import numpy as np
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.title('Predicción de condición física (FIT / NO FIT)')
st.image("fit.jpg", caption="Evaluación de condición física usando Inteligencia Artificial")

st.header('Datos de evaluación')

def user_input_features():
    age = st.number_input('Edad', min_value=1, max_value=100, value=25)
    height_cm = st.number_input('Estatura (cm)', min_value=100, max_value=220, value=170)
    weight_kg = st.number_input('Peso (kg)', min_value=30, max_value=200, value=70)
    heart_rate = st.number_input('Ritmo cardíaco', min_value=40, max_value=200, value=70)
    blood_pressure = st.number_input('Presión arterial', min_value=70, max_value=200, value=120)
    sleep_hours = st.number_input('Horas de sueño', min_value=0.0, max_value=24.0, value=7.0)
    nutrition_quality = st.number_input('Calidad de alimentación (1–10)', min_value=1.0, max_value=10.0, value=5.0)
    activity_index = st.number_input('Índice de actividad (1–10)', min_value=1.0, max_value=10.0, value=5.0)

    smokes = st.selectbox('¿Fuma?', ('No', 'Sí'))
    smokes = 1 if smokes == 'Sí' else 0

    gender = st.selectbox('Género', ('Femenino', 'Masculino'))
    gender = 1 if gender == 'Masculino' else 0

    user_data = {
        'age': age,
        'height_cm': height_cm,
        'weight_kg': weight_kg,
        'heart_rate': heart_rate,
        'blood_pressure': blood_pressure,
        'sleep_hours': sleep_hours,
        'nutrition_quality': nutrition_quality,
        'activity_index': activity_index,
        'smokes': smokes,
        'gender': gender
    }

    return pd.DataFrame(user_data, index=[0])


# =====================
# DATOS DE ENTRADA
# =====================
df = user_input_features()

# =====================
# DATASET Y MODELO
# =====================
fitness = pd.read_csv("Fitness2.csv")

X = fitness.drop(columns="is_fit")
y = fitness["is_fit"]

model = DecisionTreeClassifier(max_depth=5, criterion='gini', min_samples_leaf=10, max_features=5, random_state=0)
model.fit(X, y)

prediction = model.predict(df)[0]

# =====================
# RESULTADO
# =====================
st.subheader("Resultado de la evaluación")

if prediction == 1:
    st.success("Resultado: FIT ✅")
    st.write("La persona tiene un perfil físico saludable según el modelo.")
else:
    st.error("Resultado: NO FIT ❌")
    st.write("La persona presenta un perfil de riesgo según el modelo.")
