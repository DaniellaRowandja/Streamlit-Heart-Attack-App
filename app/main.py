import streamlit as st
import joblib
import numpy as np
import psycopg2
from dotenv import load_dotenv
import os
import uuid
from datetime import datetime
import pandas as pd
from supabase import create_client, Client

# Load model 
model = joblib.load('model/heart_attack_model.pkl')

# Connexion to Database
def add_data(records):
    # Connect to the database
    try:
        url = st.secrets["url"]
        key = st.secrets["key"]

        supabase: Client = create_client(url, key)
        supabase.table("user_inputs").insert(records).execute()

    except Exception as e:
        st.warning(f"Erreur : {e}")

# ==== STREAMLIT APP ====
st.title("Heart Attack Risk Prediction")

# Collect user input
age = st.number_input("Age", min_value=1, max_value=120, value=40)
sex = st.selectbox("Sex", ["Male", "Female"])
cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
blood_pressure = st.text_input("Blood Pressure", "163/100")
heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=220, value=80)
diabetes = st.checkbox("Diabetes")
family_history = st.checkbox("Family History")
smoking = st.checkbox("Smoking")
obesity = st.checkbox("Obesity")
alcohol_consumption = st.checkbox("Alcohol Consumption")
exercise_hours_per_week = st.number_input("Exercise Hours Per Week", min_value=0.0, max_value=30.0, value=3.0)
diet = st.selectbox("Diet", ["Poor", "Average", "Good"])
previous_heart_problems = st.checkbox("Previous Heart Problems")
medication_use = st.checkbox("Medication Use")
stress_level = st.slider("Stress Level (0-10)", 0, 10, 5)
sedentary_hours_per_day = st.number_input("Sedentary Hours Per Day", 0.0, 24.0, 8.0)
income = st.number_input("Income ($)", min_value=0.0, value=30000.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=50.0, max_value=1000.0, value=150.0)
physical_activity_days_per_week = st.slider("Physical Activity Days Per Week", 0, 7, 3)
sleep_hours_per_day = st.slider("Sleep Hours Per Day", 0, 24, 7)
country = st.text_input("Country", "France")
continent = st.selectbox("Continent", ["Europe", "Asia", "North America", "South America", "Africa", "Oceania"])
hemisphere = st.selectbox("Hemisphere", ["Northern", "Southern"])

# Convert bools to int
diabetes = int(diabetes)
family_history = int(family_history)
smoking = int(smoking)
obesity = int(obesity)
alcohol_consumption = int(alcohol_consumption)
previous_heart_problems = int(previous_heart_problems)
medication_use = int(medication_use)

# Set a dataframe
input_data = pd.DataFrame([{
    "Age": age,
    "Sex": sex,
    "Cholesterol": cholesterol,
    "Blood Pressure": blood_pressure,
    "Heart Rate": heart_rate,
    "Diabetes": diabetes,
    "Family History": family_history,
    "Smoking": smoking,
    "Obesity": obesity,
    "Alcohol Consumption": alcohol_consumption,
    "Exercise Hours Per Week": exercise_hours_per_week,
    "Diet": diet,
    "Previous Heart Problems": previous_heart_problems,
    "Medication Use": medication_use,
    "Stress Level": stress_level,
    "Sedentary Hours Per Day": sedentary_hours_per_day,
    "Income": income,
    "BMI": bmi,
    "Triglycerides": triglycerides,
    "Physical Activity Days Per Week": physical_activity_days_per_week,
    "Sleep Hours Per Day": sleep_hours_per_day,
    "Country": country,
    "Continent": continent,
    "Hemisphere": hemisphere
}])

# Predict and store
if st.button("Prédire"):

    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)

    st.success("Risque élevé" if prediction == 1 else "Risque faible")
    st.write(f"Confiance dans la prédiction de risque élevé : {proba[0][1] * 100:.2f}%")

    # Insert into PgSQL database
    input_data.columns = input_data.columns.str.strip().str.lower().str.replace(" ", "_")
    input_data["patient_id"]=str(uuid.uuid4())
    input_data["predicted_heart_attack_risk"]=int(prediction)
    input_data["created_at"]=datetime.now()

    # Insert into table 
    records = input_data.to_dict(orient="records")
    add_data(records=records)
    
    
    st.success("Données enregistrées avec succès dans la base.")

"""
    try:
        insert_query =
            INSERT INTO user_inputs (
                patient_id, age, sex, cholesterol, blood_pressure, heart_rate, diabetes,
                family_history, smoking, obesity, alcohol_consumption, exercise_hours_per_week,
                diet, previous_heart_problems, medication_use, stress_level,
                sedentary_hours_per_day, income, bmi, triglycerides,
                physical_activity_days_per_week, sleep_hours_per_day, country,
                continent, hemisphere, predicted_heart_attack_risk, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        
        data = (
            str(uuid.uuid4()), age, sex, cholesterol, blood_pressure, heart_rate, diabetes,
            family_history, smoking, obesity, alcohol_consumption, exercise_hours_per_week,
            diet, previous_heart_problems, medication_use, stress_level,
            sedentary_hours_per_day, income, bmi, triglycerides,
            physical_activity_days_per_week, sleep_hours_per_day, country,
            continent, hemisphere, int(prediction), datetime.now()
        )
        cur.execute(insert_query, data)
        conn.commit()
        st.success("Données enregistrées avec succès dans la base.")

    except Exception as e:
        st.error(f"Erreur lors de l'enregistrement dans la base : {e}")

# Close connection with database
cur.close()
conn.close()
"""