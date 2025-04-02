# Stratégie de Portefeuille Obligataire

Ce projet vise à développer et backtester des stratégies de gestion de portefeuille obligataire en utilisant différentes approches (laddering, barbell, bullet) et en analysant leur performance selon les conditions de marché et les cycles économiques.

## Objectifs du Projet

- Développer une stratégie obligataire optimisée selon différentes méthodologies
- Backtester les performances historiques des stratégies
- Évaluer l'impact des cycles économiques et des variations de taux d'intérêt
- Comparer la performance avec le Bloomberg Aggregate Bond Index

## Méthodologies Obligataires

### Laddering

La stratégie de laddering consiste à répartir les investissements dans des obligations de différentes échéances de manière échelonnée, ce qui permet :
- Une exposition diversifiée à la courbe des taux
- Un réinvestissement régulier à mesure que les obligations arrivent à échéance
- Une protection contre les fluctuations des taux d'intérêt

### Barbell

La stratégie barbell implique d'investir principalement dans des obligations à court terme et à long terme, tout en évitant les échéances intermédiaires :
- Combinaison de titres à faible duration (court terme) et à duration élevée (long terme)
- Profil de risque/rendement unique avec une sensibilité non-linéaire aux taux
- Adaptabilité aux environnements de taux volatils

### Bullet

La stratégie bullet concentre les investissements sur une échéance cible spécifique :
- Concentration des actifs autour d'une seule échéance
- Gestion précise de la duration et du risque de taux
- Optimisation pour des objectifs financiers à date fixe

## Structure du Projet

```
bond-portfolio-strategy/
├── data/                # Données obligataires et économiques
├── models/              # Implémentation des différentes stratégies
├── backtesting/         # Outils de simulation et d'analyse
├── analysis/            # Évaluation des performances et comparaisons
├── visualization/       # Graphiques et tableaux de bord
└── docs/                # Documentation et méthodologie
```

## Fonctionnalités

1. **Construction de Portefeuille**
   - Sélection d'obligations selon des critères prédéfinis
   - Allocation optimale selon chaque stratégie
   - Rééquilibrage dynamique

2. **Backtesting**
   - Simulation historique sur différentes périodes
   - Calcul des indicateurs de performance (rendement, volatilité, drawdown)
   - Analyse des métriques obligataires (duration, convexité)

3. **Analyse des Cycles Économiques**
   - Corrélation avec les indicateurs macroéconomiques
   - Impact des politiques monétaires
   - Ajustement des stratégies selon les phases de marché

## Technologies Utilisées

- **Python** : Pandas, NumPy, SciPy pour l'analyse quantitative
- **Visualisation** : Matplotlib, Seaborn, Plotly pour les graphiques
- **Modélisation** : PyPortfolioOpt, Statsmodels pour l'optimisation
- **Données** : Intégration avec des API financières pour les données obligataires

## Installation et Utilisation

```bash
# Cloner le dépôt
git clone https://github.com/Kyac99/bond-portfolio-strategy.git
cd bond-portfolio-strategy

# Installer les dépendances
pip install -r requirements.txt

# Exécuter le backtest
python run_backtest.py
```

## Planning du Projet

- **Phase 1** : Définition des critères de sélection et collecte des données (2 semaines)
- **Phase 2** : Développement des modèles de stratégie et des algorithmes de backtest (4 semaines)
- **Phase 3** : Tests, validation et ajustements (3 semaines)
- **Phase 4** : Rédaction du rapport final et mise en production (3 semaines)

## Licence

Ce projet est sous licence MIT.
