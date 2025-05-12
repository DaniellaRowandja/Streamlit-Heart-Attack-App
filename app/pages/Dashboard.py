import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
from dotenv import load_dotenv
import os

# === CONFIGURATION ===
load_dotenv()
DATABASE_URL = st.secrets["SUPABASE_DB_URL"]

# Connexion to Database
def load_user_data():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        query = "SELECT * FROM user_inputs"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.warning(f"Erreur de connexion : {e}")
        return pd.DataFrame()

# === DASHBOARD ===
st.title("📊 Dashboard - Heart Attack Risk")

tab1, tab2 = st.tabs(["Données du Dataset Kaggle", "Données Collectées des Utilisateurs"])

with tab1:
    st.header("📁 Données Originales (Kaggle)")

    try:
        df_kaggle = pd.read_csv("datasets/heart_attack_prediction_dataset.csv")
        st.write("Aperçu du dataset :", df_kaggle.head())

        st.subheader("Répartition des risques d'attaque cardiaque")
        fig1, ax1 = plt.subplots()
        sns.countplot(x="Heart Attack Risk", data=df_kaggle, ax=ax1)
        st.pyplot(fig1)

        st.subheader("Corrélation entre les variables")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_kaggle.corr(numeric_only=True), cmap="coolwarm", annot=False, ax=ax2)
        st.pyplot(fig2)

        st.subheader("Distribution du cholestérol")
        fig3, ax3 = plt.subplots()
        sns.histplot(df_kaggle["Cholesterol"], kde=True, ax=ax3)
        st.pyplot(fig3)

    except Exception as e:
        st.error(f"Erreur de chargement des données Kaggle : {e}")

with tab2:
    st.header("📥 Données des Utilisateurs (Base Supabase)")

    df_user = load_user_data()

    if df_user.empty:
        st.info("Aucune donnée utilisateur disponible.")
    else:

        col1, col2, col3 = st.columns(3)
        col1.metric("Nombre de prédictions", len(df_user))
        col2.metric("Risque moyen", round(df_user["predicted_heart_attack_risk"].mean(), 2))
        col3.metric("Âge moyen", int(df_user["age"].mean()))

        st.write("Aperçu des données utilisateurs :", df_user.tail())

        st.subheader("Répartition des prédictions")
        fig4, ax4 = plt.subplots()
        sns.countplot(x="predicted_heart_attack_risk", data=df_user, ax=ax4)
        ax4.set_xticklabels(["Faible Risque", "Haut Risque"])
        st.pyplot(fig4)

        st.subheader("Distribution de l'âge")
        fig5, ax5 = plt.subplots()
        sns.histplot(df_user["age"], bins=20, kde=True, ax=ax5)
        st.pyplot(fig5)

        st.subheader("Comparaison Cholestérol - Prédiction")
        fig6, ax6 = plt.subplots()
        sns.boxplot(x="predicted_heart_attack_risk", y="cholesterol", data=df_user, ax=ax6)
        st.pyplot(fig6)

        if "continent" in df_user.columns:
            st.subheader("Moyenne des prédictions par continent")
            continent_avg = df_user.groupby("continent")["predicted_heart_attack_risk"].mean().sort_values(ascending=False)
            st.bar_chart(continent_avg)
