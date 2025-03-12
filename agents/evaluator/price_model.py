#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Agent Évaluateur - Modèle de prédiction de prix

Ce module contient la classe principale pour la prédiction des prix
de revente d'articles de maroquinerie de luxe.
"""

import os
import json
import logging
import pickle
from typing import Dict, List, Tuple, Union, Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PricePredictor:
    """
    Classe pour la prédiction des prix de revente de produits de luxe
    """
    
    def __init__(self, model_type: str = "xgboost"):
        """
        Initialise le prédicteur de prix
        
        Args:
            model_type: Type de modèle à utiliser ("xgboost", "random_forest", "gradient_boosting")
        """
        self.model_type = model_type
        self.model = None
        self.feature_encoders = {}
        self.scaler = None
        self.feature_importance = None
        self.metrics = None
        self.categorical_features = [
            'brand', 'model', 'material', 'color', 'condition',
            'has_receipt', 'has_dustbag', 'has_box', 'is_limited_edition'
        ]
        self.numerical_features = [
            'age_years', 'original_price', 'avg_market_price',
            'popularity_score', 'seasonality_factor'
        ]
    
    def _initialize_model(self):
        """Initialise le modèle de ML en fonction du type spécifié"""
        if self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            )
        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}")
    
    def preprocess_data(self, data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Prétraite les données pour l'entraînement ou la prédiction
        
        Args:
            data: Données à prétraiter (dictionnaire ou DataFrame)
            
        Returns:
            DataFrame prétraité prêt pour l'entraînement ou la prédiction
        """
        # Convertir le dictionnaire en DataFrame si nécessaire
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Copie pour éviter les modifications sur les données d'origine
        df = data.copy()
        
        # Traitement des caractéristiques catégorielles
        for feature in self.categorical_features:
            if feature in df.columns:
                if feature in self.feature_encoders:
                    # Utiliser l'encodeur existant pour la transformation
                    encoder = self.feature_encoders[feature]
                    # Gérer les nouvelles catégories non vues pendant l'entraînement
                    new_categories = set(df[feature].unique()) - set(encoder.classes_)
                    if new_categories:
                        logger.warning(f"Nouvelles catégories trouvées dans {feature}: {new_categories}")
                        # Remplacer par "unknown" ou une valeur par défaut
                        df.loc[df[feature].isin(new_categories), feature] = "unknown"
                    
                    # Transformer en one-hot encoding
                    encoded = pd.get_dummies(df[feature], prefix=feature)
                    # Ajouter des colonnes manquantes si nécessaire
                    expected_columns = [f"{feature}_{cat}" for cat in encoder.classes_]
                    for col in expected_columns:
                        if col not in encoded.columns:
                            encoded[col] = 0
                    
                    # Supprimer les colonnes inattendues
                    extra_columns = set(encoded.columns) - set(expected_columns)
                    if extra_columns:
                        encoded = encoded.drop(columns=list(extra_columns))
                    
                    # Remplacer la colonne originale par les colonnes encodées
                    df = df.drop(columns=[feature])
                    df = pd.concat([df, encoded[expected_columns]], axis=1)
        
        # Traitement des caractéristiques numériques
        for feature in self.numerical_features:
            if feature in df.columns:
                # Remplacer les valeurs manquantes par la moyenne
                if df[feature].isna().any():
                    if self.scaler is not None and feature in self.scaler.mean_:
                        df[feature] = df[feature].fillna(self.scaler.mean_[feature])
                    else:
                        df[feature] = df[feature].fillna(df[feature].mean())
                
                # Appliquer la normalisation si disponible
                if self.scaler is not None:
                    df[feature] = self.scaler.transform(df[[feature]])[0]
        
        return df
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Entraîne le modèle de prédiction de prix
        
        Args:
            X_train: DataFrame contenant les caractéristiques d'entraînement
            y_train: Série contenant les prix cibles
            
        Returns:
            Métriques d'entraînement
        """
        # Initialiser le modèle
        self._initialize_model()
        
        # Prétraiter les données
        X_processed = self.preprocess_data(X_train)
        
        # Entraîner le modèle
        self.model.fit(X_processed, y_train)
        
        # Calculer les métriques d'entraînement
        predictions = self.model.predict(X_processed)
        mse = np.mean((predictions - y_train) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_train))
        r2 = 1 - (np.sum((y_train - predictions) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2))
        
        # Stocker les métriques
        self.metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        
        # Calculer l'importance des caractéristiques si disponible
        if hasattr(self.model, "feature_importances_"):
            feature_names = X_processed.columns
            self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
            
            # Trier par importance décroissante
            self.feature_importance = {k: v for k, v in sorted(
                self.feature_importance.items(), key=lambda item: item[1], reverse=True
            )}
        
        logger.info(f"Modèle entraîné avec RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        return self.metrics
    
    def predict(self, item_data: Dict) -> Dict:
        """
        Prédit le prix de revente pour un article
        
        Args:
            item_data: Dictionnaire contenant les caractéristiques de l'article
            
        Returns:
            Dictionnaire contenant le prix prédit et l'intervalle de confiance
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné ou chargé")
        
        # Prétraiter les données
        X = self.preprocess_data(item_data)
        
        # Prédiction
        predicted_price = self.model.predict(X)[0]
        
        # Calculer l'intervalle de confiance (approche simple)
        # Dans un cas réel, on utiliserait des méthodes plus sophistiquées
        if self.metrics:
            confidence_width = 1.96 * self.metrics["rmse"]  # Intervalle de confiance à 95%
            lower_bound = max(0, predicted_price - confidence_width)
            upper_bound = predicted_price + confidence_width
        else:
            # Valeur par défaut si les métriques ne sont pas disponibles
            lower_bound = 0.8 * predicted_price
            upper_bound = 1.2 * predicted_price
        
        result = {
            "price": round(predicted_price, 2),
            "confidence_interval": (round(lower_bound, 2), round(upper_bound, 2)),
            "confidence_width_percent": round((upper_bound - lower_bound) / predicted_price * 100, 1)
        }
        
        # Ajouter les facteurs influençant le prix si l'importance des caractéristiques est disponible
        if self.feature_importance:
            # Prendre les 5 caractéristiques les plus importantes
            top_features = list(self.feature_importance.keys())[:5]
            result["top_factors"] = top_features
        
        return result
    
    def save_model(self, model_path: str):
        """
        Sauvegarde le modèle entraîné
        
        Args:
            model_path: Chemin où sauvegarder le modèle
        """
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder")
        
        # Créer le répertoire si nécessaire
        os.makedirs(model_path, exist_ok=True)
        
        # Sauvegarder le modèle
        with open(os.path.join(model_path, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        
        # Sauvegarder les encodeurs et le scaler
        with open(os.path.join(model_path, "encoders.pkl"), "wb") as f:
            pickle.dump(self.feature_encoders, f)
        
        if self.scaler is not None:
            with open(os.path.join(model_path, "scaler.pkl"), "wb") as f:
                pickle.dump(self.scaler, f)
        
        # Sauvegarder les métriques et l'importance des caractéristiques
        metadata = {
            "model_type": self.model_type,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features
        }
        
        with open(os.path.join(model_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Modèle sauvegardé dans {model_path}")
    
    def load_model(self, model_path: str):
        """
        Charge un modèle entraîné
        
        Args:
            model_path: Chemin du modèle à charger
        """
        # Charger le modèle
        with open(os.path.join(model_path, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        
        # Charger les encodeurs et le scaler
        with open(os.path.join(model_path, "encoders.pkl"), "rb") as f:
            self.feature_encoders = pickle.load(f)
        
        scaler_path = os.path.join(model_path, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        
        # Charger les métadonnées
        with open(os.path.join(model_path, "metadata.json"), "r") as f:
            metadata = json.load(f)
            self.model_type = metadata.get("model_type", self.model_type)
            self.metrics = metadata.get("metrics")
            self.feature_importance = metadata.get("feature_importance")
            self.categorical_features = metadata.get("categorical_features", self.categorical_features)
            self.numerical_features = metadata.get("numerical_features", self.numerical_features)
        
        logger.info(f"Modèle chargé depuis {model_path}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer un exemple de données
    sample_data = {
        'brand': 'Louis Vuitton',
        'model': 'Neverfull MM',
        'material': 'Monogram Canvas',
        'color': 'Brown',
        'condition': 'Very good',
        'age_years': 3,
        'original_price': 1200,
        'has_receipt': True,
        'has_dustbag': True,
        'has_box': False,
        'is_limited_edition': False,
        'avg_market_price': 950,
        'popularity_score': 8.5,
        'seasonality_factor': 1.02
    }
    
    # Instancier le prédicteur
    predictor = PricePredictor(model_type="xgboost")
    
    # Dans un cas réel, on chargerait un modèle entraîné
    # predictor.load_model("models/price_predictor/latest")
    
    # Comme nous n'avons pas de modèle entraîné, simulons un modèle simple
    predictor.model = lambda x: [0.7 * sample_data['original_price'] * sample_data['seasonality_factor']]
    predictor.model.predict = predictor.model
    predictor.metrics = {"rmse": 50}
    
    # Faire une prédiction
    prediction = predictor.predict(sample_data)
    print(f"Prix estimé: {prediction['price']} €")
    print(f"Intervalle de confiance: {prediction['confidence_interval']}")
