# Prédiction d’Attaque Cardiaque avec Machine Learning

Ce projet a pour objectif de prédire le risque d’attaque cardiaque à partir de données médicales en utilisant des algorithmes de machine learning supervisé. Il comprend une analyse exploratoire des données (EDA), l'entraînement de modèles de classification, ainsi qu'une application web interactive développée avec Streamlit.

---

## Contenu du projet

- `EDA_HeartAttackPrediction.ipynb` : Notebook Jupyter contenant l’analyse exploratoire, le traitement des données, l'entraînement et l’évaluation des modèles.
- `heart_attack_model.pkl` : Modèle entraîné (Random Forest).
- `scaler.pkl` : StandardScaler utilisé pour la normalisation des données.
- `app.py` : Application Streamlit permettant de tester le modèle de manière interactive.
- `README.md` : Ce fichier.

---

## Description du dataset

- **Source** : [Kaggle - Heart Attack Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset)
- **Objectif** : Prédire la variable `Heart Attack Risk` (0 = absence de risque, 1 = présence d’un risque d’attaque cardiaque)
- **Variables principales** :
  - Âge, sexe, tension artérielle, cholestérol, fréquence cardiaque maximale, dépression ST, etc.

---

## Installation

1. Cloner le dépôt ou télécharger les fichiers nécessaires.
2. Installer les dépendances requises :

```bash
pip install -r requirements.txt
```

3. Lancer le notebook d’analyse :

```bash
jupyter notebook EDA_ML_HeartAttackPrediction.ipynb
```

4. Lancer l’application Streamlit :

```bash
streamlit run main.py
```

---

## Fonctionnement de l'application

L'utilisateur peut saisir différentes données de santé (âge, tension, cholestérol, etc.), et l’application retourne une prédiction sur le risque d’attaque cardiaque à l’aide du modèle préalablement entraîné.

---

## Modèles utilisés

- Modèle par défaut : Logistic Regression
- Autres modèles testables dans le notebook : Random Forest, SVM, KNN, etc.

---

## Améliorations possibles

- Comparaison de plusieurs algorithmes et optimisation des hyperparamètres
- Ajout de tests unitaires et validation croisée

---

## Auteur

Projet réalisé dans le cadre d'une étude sur la prédiction des risques cardiovasculaires à l’aide de données ouvertes.  
Contact : [Rowandja Daniella - daniellarowandja@gmail.com]
