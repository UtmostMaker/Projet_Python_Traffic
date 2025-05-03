# Prédiction du Volume de Trafic Inter-États Métropolitain

## Description

Ce projet vise à prédire le volume de trafic horaire sur l'autoroute I-94 Interstate entre Minneapolis et St Paul, Minnesota. L'objectif est de construire un modèle de régression [1] en utilisant des données historiques de trafic ainsi que des caractéristiques météorologiques et de jours fériés pour prévoir le flux de trafic. Ce projet est développé dans le cadre d'un cours de Python, axé sur l'analyse de données, la construction de pipelines de machine learning et le déploiement de modèles (à venir).

## Jeu de Données [1]

*   **Source :** Kaggle - [Metro Interstate Traffic Volume Dataset](https://www.kaggle.com/datasets/anshtanwar/metro-interstate-traffic-volume)
*   **Contenu :** Environ 48 000 enregistrements horaires couvrant la période de 2012 à 2018 [3].
*   **Caractéristiques :** Inclut des informations de date/heure, les conditions météorologiques (température, pluie, neige, nuages), des descriptions d'événements météo et des indicateurs de jours fériés [3].
*   **Variable Cible :** `traffic_volume` (numérique, représentant les véhicules par heure) [3].
*   **Fichier :** `data/Metro_Interstate_Traffic_Volume.csv` [3]

## Structure du Projet

Projet_Python_Traffic/
│
├── .venv/ # Environnement virtuel (ignoré par Git)
├── .git/ # Données du dépôt Git (caché)
├── .gitignore # Spécifie les fichiers intentionnellement non suivis (ex: .venv, gros modèles)
├── README.md # Ce fichier de documentation
├── requirements.txt # Dépendances Python du projet
│
├── data/ # Données brutes et traitées
│ └── Metro_Interstate_Traffic_Volume.csv
│
├── notebooks/ # Notebooks Jupyter pour l'analyse et l'entraînement du modèle
│ └── 01_entrainement_modele_trafic.ipynb
│
├── models/ # Artefacts des modèles entraînés (fichiers .pkl)
│ ├── traffic_prediction_RandomForest_tuned.pkl # Modèle final optimisé
│ └── traffic_prediction_RandomForest_untuned.pkl # Meilleur modèle non optimisé (pour comparaison)
│
└── app/ # Code source pour l'application web Streamlit
├── app.py # Script principal de l'application (À créer)
└── traffic_prediction_RandomForest_tuned.pkl # Copie du modèle final pour l'app



## Configuration et Installation

1.  **Cloner le Dépôt :**
    ```
    git clone https://github.com/UtmostMaker/Projet_Python_Traffic.git
    cd Projet_Python_Traffic
    ```

2.  **Créer et Activer l'Environnement Virtuel :**
    *   **Linux/macOS :**
        ```
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    *   **Windows (cmd) :**
        ```
        python -m venv .venv
        .venv\Scripts\activate
        ```
    *   **Windows (PowerShell) :**
        ```
        python -m venv .venv
        .venv\Scripts\Activate.ps1
        ```

3.  **Installer les Dépendances :** Assurez-vous que votre environnement virtuel est actif et installez tous les paquets requis.
    ```
    pip install -r requirements.txt
    ```

## Flux de Travail et Processus de Modélisation (Résumé)

L'analyse et le développement du modèle sont détaillés dans le notebook `notebooks/01_entrainement_modele_trafic.ipynb` [2]. Les étapes clés comprennent :

1.  **Chargement des Données :** Importation du jeu de données avec Pandas [2].
2.  **Analyse Exploratoire des Données (EDA) :** Compréhension des distributions, corrélations et identification des motifs liés au temps, à la météo et aux jours fériés. Visualisations créées avec Matplotlib et Seaborn [2].
3.  **Nettoyage des Données :** Traitement des valeurs manquantes et des doublons potentiels (détails dans le notebook) [2].
4.  **Ingénierie de caractéristiques (Feature Engineering) :** Extraction de caractéristiques à partir des colonnes date/heure (heure, jour de la semaine, mois, année). Encodage potentiel des jours fériés ou événements spéciaux [2].
5.  **Prétraitement :**
    *   Séparation des données en ensembles d'entraînement et de test [2].
    *   Mise à l'échelle des caractéristiques numériques (ex: avec `StandardScaler`) [2].
    *   Encodage des caractéristiques catégorielles (ex: avec `OneHotEncoder` ou `TargetEncoder` pour les descriptions météo) [2].
    *   Création d'un pipeline de prétraitement avec `ColumnTransformer` [2].
6.  **Sélection de Modèle :** Construction de pipelines complets (prétraitement + modèle) et évaluation de plusieurs modèles de régression (ex: Ridge, SVR, GradientBoosting, RandomForest) en utilisant la validation croisée et la RMSE comme métrique principale. RandomForestRegressor a montré les meilleures performances initiales [2].
7.  **Optimisation d'Hyperparamètres :** Optimisation du modèle le plus performant (RandomForestRegressor) avec `GridSearchCV` pour trouver la meilleure combinaison d'hyperparamètres basée sur la RMSE en validation croisée [2].
8.  **Évaluation :** Évaluation de la performance du modèle final optimisé sur l'ensemble de test non utilisé, en utilisant la RMSE et le score R² [2].
9.  **Sauvegarde du Modèle :** Sérialisation (enregistrement) du pipeline final optimisé (prétraitement + meilleur modèle optimisé) en utilisant `joblib` dans le répertoire `models/` (`traffic_prediction_RandomForest_tuned.pkl`) [1][2].

## Utilisation

*   **Relancer l'Analyse/Entraînement :**
    1.  Activez l'environnement virtuel (`source .venv/bin/activate` ou équivalent).
    2.  Lancez Jupyter Notebook, JupyterLab, ou ouvrez le projet dans VS Code.
    3.  Ouvrez et exécutez les cellules dans `notebooks/01_entrainement_modele_trafic.ipynb`. Cela effectuera l'ensemble du flux de travail et régénérera les fichiers de modèle dans le répertoire `models/`.
*   **Utiliser le Modèle Entraîné :** Le modèle final, prêt à l'emploi, se trouve à l'emplacement `models/traffic_prediction_RandomForest_tuned.pkl`. Le script `app/app.py` (lorsqu'il sera créé) chargera ce modèle (ou sa copie dans `app/`) pour effectuer des prédictions.

## Gestion de Version (GitHub)

Ce projet est versionné avec Git et hébergé sur GitHub.
*   Le fichier `.gitignore` est configuré pour exclure l'environnement virtuel (`.venv/`) et les fichiers de modèle volumineux (`*.pkl` dans `models/`) afin de respecter les limites de taille de fichier de GitHub et de maintenir un dépôt propre.
*   Les modifications sont enregistrées (commit) régulièrement. L'historique a pu être réécrit (`git-filter-repo`, `git push --force`) lors de la configuration initiale pour supprimer des fichiers volumineux accidentellement commités.

## Prochaines Étapes : Application Web [1]

*   **Objectif :** Développer une application web simple utilisant Streamlit pour permettre aux utilisateurs de saisir des caractéristiques pertinentes (comme la date, l'heure, les conditions météo) et d'obtenir une prédiction du volume de trafic basée sur le modèle entraîné.
*   **Emplacement :** Le code de l'application se trouvera dans le répertoire `app/`, principalement dans `app.py`.
*   **Fonctionnalités :**
    *   Charger le modèle pré-entraîné (`app/traffic_prediction_RandomForest_tuned.pkl`).
    *   Fournir des éléments d'interface utilisateur (curseurs, listes déroulantes, sélecteurs de date) pour les caractéristiques d'entrée [1].
    *   Prétraiter l'entrée utilisateur pour correspondre au format attendu par le pipeline du modèle.
    *   Afficher clairement le volume de trafic prédit [1].

## Technologies Utilisées

*   Python 3.x
*   Jupyter Notebook / VS Code
*   Git / GitHub
*   Pandas
*   NumPy
*   Scikit-learn
*   Category Encoders
*   Matplotlib
*   Seaborn
*   Joblib
*   Streamlit (pour la future application web)


