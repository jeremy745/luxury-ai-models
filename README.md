# Luxury AI Models

Ce dépôt contient les modèles d'intelligence artificielle et les agents utilisés pour la prédiction des prix de revente d'articles de maroquinerie de luxe.

## 🌟 Vue d'ensemble

Ce projet utilise une architecture multi-agents pour analyser et prédire les prix de revente des articles de maroquinerie de luxe. Les agents sont spécialisés pour différentes tâches et travaillent ensemble pour fournir des estimations précises.

## 🧩 Agents IA

### Agent Extracteur
- Analyse les descriptions textuelles et les images
- Extrait les attributs clés : marque, modèle, matériau, année, état, etc.
- Normalise les données extraites

### Agent Comparateur
- Recherche des articles similaires dans la base de données
- Calcule les scores de similarité
- Identifie les produits les plus comparables

### Agent Évaluateur
- Prédit les prix de revente basés sur les attributs et les données historiques
- Calcule des intervalles de confiance
- Ajuste les prédictions en fonction des ventes récentes

### Agent Tendances
- Analyse les tendances du marché
- Détecte les facteurs saisonniers
- Surveille l'évolution de la popularité des marques et modèles

## 📂 Structure du dépôt

```
luxury-ai-models/
├── agents/
│   ├── extractor/             # Agent d'extraction d'attributs
│   │   ├── text_analyzer.py   # Analyse des descriptions
│   │   └── image_analyzer.py  # Analyse des images
│   │
│   ├── comparator/            # Agent de comparaison
│   │   ├── similarity.py      # Calcul de similarité
│   │   └── search.py          # Recherche d'articles similaires
│   │
│   ├── evaluator/             # Agent d'évaluation
│   │   ├── price_model.py     # Modèle de prédiction de prix
│   │   └── confidence.py      # Calcul d'intervalles de confiance
│   │
│   └── trends/                # Agent de tendances
│       ├── market_analyzer.py # Analyse du marché
│       └── popularity.py      # Suivi de popularité
│
├── models/                    # Modèles entraînés
│   ├── price_predictor/       # Modèle principal de prédiction
│   ├── attribute_extractor/   # Modèle d'extraction d'attributs
│   └── trend_analyzer/        # Modèle d'analyse de tendances
│
├── training/                  # Scripts d'entraînement
│   ├── price_model/           # Entraînement du modèle de prix
│   ├── feature_extraction/    # Entraînement des extracteurs
│   └── evaluation/            # Évaluation des modèles
│
├── notebooks/                 # Jupyter notebooks
├── tests/                     # Tests unitaires et d'intégration
└── data/                      # Données d'exemple et de test
```

## 🚀 Installation

```bash
# Cloner le dépôt
git clone https://github.com/jeremy745/luxury-ai-models.git
cd luxury-ai-models

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Pour Linux/Mac
# ou
venv\Scripts\activate     # Pour Windows

# Installer les dépendances
pip install -r requirements.txt
```

## 💻 Utilisation

```python
# Exemple d'utilisation (à implémenter)
from agents.evaluator.price_model import PricePredictor

# Instancier le prédicteur
predictor = PricePredictor()

# Charger le modèle
predictor.load_model('models/price_predictor/latest')

# Faire une prédiction
item_data = {
    'brand': 'Louis Vuitton',
    'model': 'Neverfull MM',
    'material': 'Monogram Canvas',
    'year': 2019,
    'condition': 'Très bon état'
}

prediction = predictor.predict(item_data)
print(f"Prix estimé: {prediction['price']} €")
print(f"Intervalle de confiance: {prediction['confidence_interval']}")
```

## 🔧 Architecture du système d'agents

Les agents sont conçus pour fonctionner ensemble dans un pipeline:

1. L'**Agent Extracteur** traite d'abord les données brutes
2. L'**Agent Comparateur** identifie des produits similaires
3. L'**Agent Évaluateur** calcule un prix estimé
4. L'**Agent Tendances** ajuste l'estimation selon les tendances actuelles

Les agents communiquent via une architecture de messages, permettant un traitement asynchrone et une scalabilité horizontale.

## 📊 Modèles de prédiction

Ce projet utilise plusieurs approches de modélisation:

- **Modèle de base**: XGBoost pour la prédiction initiale
- **Modèle avancé**: Réseau de neurones profond avec des incorporations pour les caractéristiques catégorielles
- **Modèle d'ensemble**: Combinaison de plusieurs approches pour améliorer la robustesse

## 🛠️ Développement

### Ajouter un nouveau modèle

1. Créez un nouveau dossier dans `models/`
2. Implémentez le modèle en suivant l'interface commune
3. Ajoutez les scripts d'entraînement dans `training/`
4. Mettez à jour les tests et la documentation

## 📝 Roadmap

- [ ] Implémentation de l'Agent Extracteur
- [ ] Implémentation de l'Agent Comparateur  
- [ ] Implémentation de l'Agent Évaluateur
- [ ] Implémentation de l'Agent Tendances
- [ ] Intégration des agents dans un pipeline cohérent
- [ ] Déploiement des modèles en production
