# Mini-projet : Planification robuste sur grille – A* + Chaînes de Markov

**Auteur** : Fatima-zahra Massane  
**Filière** : SDIA-1  
**Encadrant** : Prof. M. Messtari  
**Date** : Mars 2026  
**Institution** : ENSET

## Description du projet

Ce mini-projet implémente une approche hybride pour la planification de chemin dans un environnement incertain :

- **Partie déterministe** : algorithme **A*** (et variantes UCS, Greedy, Weighted A*) avec heuristique Manhattan admissible et cohérente.
- **Partie stochastique** : modélisation des incertitudes d’action via **chaînes de Markov** à temps discret :
  - Politique déterministe extraite du chemin A*
  - Matrice de transition P avec probabilité 1-ε vers l’action voulue et ε/2 vers chaque latéral
  - États absorbants : GOAL et FAIL
  - Calcul exact des probabilités d’absorption (matrice fondamentale N = (I-Q)⁻¹)
  - Simulation Monte-Carlo (5000 trajectoires) pour validation

Objectif principal : montrer que même un plan optimal déterministe devient très fragile dès que l’incertitude (ε) augmente, et quantifier cette dégradation (probabilité réelle d’atteindre le but, temps moyen d’absorption).

## Structure du dépôt
Mini_projet/

 --> astar.py               # A*, UCS, Greedy, Weighted A*
 --> grid.py                # Génération grilles easy/medium/hard + visualisation basique
 --> markov.py              # Politique du chemin, matrice P, absorption, simulation MC, analyse classes
 --> experiments.py         # Exécution automatique des 4 expériences (E1 à E4) + CSV
 --> visualizations.py      # Heatmaps P, courbes π(n), histogrammes, graphes NetworkX
 --> main.py                # Lance tout (expériences + visualisations clés)
 --> Visu.ipynb             # Notebook Jupyter complet pour présentation (tableaux stylés + toutes les figures)
 --> results/               # Résultats CSV (experiment_1, _2, _3, _4)
 --> figures/               # Toutes les images générées (chemins, heatmaps, courbes, histogrammes...)
 --> README.md              # Ce fichier
##
## Prérequis

- Python 3.8+
- Bibliothèques :
  ```bash
  pip install numpy matplotlib pandas seaborn networkx

## Comment exécuter
### 1.clone le projet

git clone <URL de ton dépôt>
cd Mini_projet

### 2.Lancer l’ensemble du projet

python main.py


## Ouvrir le notebook de présentation
jupyter notebook Visu.ipynb
