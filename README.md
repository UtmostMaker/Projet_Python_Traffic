# 🚦 Projet de Prédiction du Volume de Trafic Autoroutier

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-blueviolet)

## 📖 Table des Matières
- [Objectif du Projet](#🎯-objectif-du-projet)
- [Jeu de Données](#📊-jeu-de-données)
- [Architecture du Projet](#📁-architecture-du-projet)
- [Méthodologie](#🔧-méthodologie)
- [Résultats](#📈-résultats)
- [Déploiement](#🚀-déploiement)
- [Défis Rencontrés](#⚠️-défis-rencontrés)
- [Compétences Acquises](#💡-compétences-acquises)
- [Perspectives](#🔮-perspectives)

## 🎯 Objectif du Projet
Développer un modèle de prédiction du volume horaire de trafic sur l'autoroute I-94 (Minneapolis-St Paul) en utilisant :
- Données historiques (2012-2018)
- Variables météorologiques
- Informations temporelles
- Jours fériés

## 📊 Jeu de Données
**Source :** [Kaggle - Metro Interstate Traffic Volume](https://www.kaggle.com/datasets/anshtanwar/metro-interstate-traffic-volume)  
**Caractéristiques clés :**
- 48 000 enregistrements horaires
- 10 variables explicatives
- Variables météorologiques détaillées
- Marqueurs temporels (heure, jour, mois, année)

## 📁 Architecture du Projet


```

.
├── app
│   ├── app.py
│   └── requirements.txt
├── arborescence.txt
├── data
│   └── Metro_Interstate_Traffic_Volume.csv
├── .git
│   ├── COMMIT_EDITMSG
│   ├── config
│   ├── description
│   ├── filter-repo
│   ├── HEAD
│   ├── hooks
│   ├── index
│   ├── info
│   ├── logs
│   ├── objects
│   ├── ORIG_HEAD
│   ├── packed-refs
│   └── refs
├── .gitignore
├── models
│   ├── traffic_prediction_RandomForest_tuned.pkl
│   └── traffic_prediction_RandomForest_untuned.pkl
├── notebooks
│   ├── 00_exploration_datas_trafic.ipynb
│   └── 01_entrainement_modele_trafic.ipynb
├── README.md
├── requirements.txt
├── src
│   ├── custom_transformers.py
│   └── __pycache__
└── .venv
    ├── bin
    ├── etc
    ├── .gitignore
    ├── include
    ├── lib
    ├── lib64 -> lib
    ├── pyvenv.cfg
    └── share
```


## 🔧 Méthodologie
1. **Prétraitement des Données**
   - Conversion des températures (Kelvin → Celsius)
   - Feature engineering : pics horaires, saisons, précipitations
   - Encodage personnalisé des variables catégorielles

2. **Modélisation**
   - Comparaison de 5 algorithmes (Random Forest, Gradient Boosting, SVR, etc.)
   - Optimisation hyperparamétrique avec GridSearchCV
   - Validation croisée (3 folds)

3. **Déploiement**
   - Interface utilisateur avec Streamlit
   - Prédictions en temps réel
   - Visualisation interactive des résultats

## 📈 Résultats
**Modèle Final :** Random Forest Optimisé  
**Performances :**
- RMSE : 528.44 véhicules/heure
- R² : 0.928
- Temps d'exécution : < 1 seconde/prédiction

**Comparaison des Modèles :**
| Modèle            | RMSE    | R²     |
|-------------------|---------|--------|
| Random Forest     | 528.44  | 0.928  |
| Gradient Boosting | 593.16  | 0.909  |
| Régression Linéaire | 1574.15 | 0.360 |

## 🚀 Déploiement
**Exécuter l'application :**

###Configuration de l'environnement

python -m venv .venv
source .venv/bin/activate # Linux/Mac

.venv\Scripts\activate # Windows

pip install -r requirements.txt

###Lancement de l'application

cd app
streamlit run app.py


## ⚠️ Défis Rencontrés

1. **Implémentation d'Encodages Personnalisés**  
   - **FrequencyEncoder** : Création d'un transformateur compatible scikit-learn (`BaseEstimator`, `TransformerMixin`) pour encoder les catégories par leur fréquence, avec gestion des valeurs inconnues (remplacement par 0).  
   - **Compatibilité Pipeline** : Intégration complexe des encodages spécifiques (*OneHot* pour météo/jours fériés, *Target Encoding* pour pics horaires) tout en évitant les fuites de données.

2. **Architecture du Projet**  
   - Structuration en modules (`src`, `app`, `models`, `notebooks`) avec gestion des imports relatifs.  
   - Résolution des conflits de chemins entre l'entraînement (notebooks) et le déploiement (app Streamlit), nécessitant des ajustements de `sys.path`.

3. **Traitement des Données Temporelles**  
   - Découpage chronologique des données (2012-2018) pour éviter les fuites temporelles, contrairement à un split aléatoire classique.  
   - Extraction manuelle des features cycliques (heure, jour, saison) pour capturer les motifs récurrents.

4. **Optimisation des Performances**  
   - **GridSearchCV** sur RandomForest : 3h d'exécution pour explorer 36 combinaisons d'hyperparamètres (`n_estimators`, `max_depth`, `min_samples_split`).  
   - Compromis entre précision (RMSE=528) et temps de prédiction (<1s) pour l'usage en temps réel.

5. **Débogage Multi-Outils**  
   - Utilisation combinée du debugger VS Code, des prints stratégiques et des erreurs Streamlit pour traquer les incompatibilités de données.

## 💡 Compétences Acquises

### **Machine Learning Avancé**
- **Feature Engineering** : Création de variables métier (pics horaires, statut précipitations)  
- **Pipelines Modularisés** : Combinaison de `ColumnTransformer`, `StandardScaler` et encodeurs personnalisés  
- **Optimisation** : Maîtrise de `GridSearchCV` et interprétation des hyperparamètres (`max_depth=10`, `min_samples_split=5`)  
- **Évaluation** : Calcul et interprétation de RMSE (528 véhicules/heure) et R² (0.928)

### **Développement Logiciel**
- **POO en Python** : Implémentation de classes compatibles scikit-learn  
- **Gestion de Projet** : Architecture modulaire, gestion des dépendances avec `requirements.txt`  
- **Déploiement** : Création d'une interface utilisateur avec Streamlit et gestion des chemins relatifs

### **Outils Professionnels**
- **VS Code** : Maîtrise du debugger, gestion multi-fenêtres (Jupyter + app)  
- **Git** : Collaboration via commits atomiques et résolution de conflits  
- **Visualisation** : Création de dashboards interactifs avec `matplotlib`/`seaborn`

### **Modèles Expérimentés**
- **RandomForest** : Meilleures performances (RMSE=528) grâce au réglage fin  
- **Gradient Boosting** : Deuxième meilleur modèle (RMSE=593), plus lent à entraîner  
- **Régression Linéaire** : Performances médiocres (RMSE=1574) montrant la non-linéarité des données
- **SVR** : Modèle entraîné pour comparaison, mais dont les performances (RMSE=1777) restent en retrait sur ce problème.

## 🔮 Perspectives

### 🚀 Au-Delà du Cours Initial
Ce projet, bien que réalisé dans le cadre d'un cours d'introduction à Python, a permis d'explorer des concepts normalement réservés à des niveaux avancés grâce à :

**Collaboration avec IA** :  
- Génération de code boilerplate pour les transformers  
- Décryptage des erreurs obscures   
- Optimisation des requêtes de recherche pour résoudre des bugs spécifiques  

**Apprentissage Accéléré** :  
- Reverse-engineering de solutions professionnelles via l'étude de code GitHub  
- Adaptation de tutoriels (ex: déploiement de modèles) au contexte pédagogique  

**Pensée Systémique** :  
- Intégration fluide entre composants (notebooks → modèle → app)  
- Gestion des contraintes réelles (mémoire, temps CPU, ergonomie utilisateur)  

Cette expérience démontre comment l'IA peut servir de multiplicateur de compétences, permettant à des débutants de réaliser des projets normalement hors de portée tout en développant une compréhension des mécanismes sous-jacents.

---

**Étudiants :** [Brice, Anaïs, Raoul ]  
**Encadrant :** [Alexis BOGROFF]  
**Date de Livraison :** 18 Mai 2025  

