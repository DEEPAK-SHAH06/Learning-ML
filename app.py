import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler_heart.pkl')
expected_colums = joblib.load("columns_heart.pkl")


st.title("Heart Disease Prediction")
st.markdown("Enter the following details to predict the likelihood of heart disease:")

age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("SEX",['M', 'F'])
chest_pain_type = st.selectbox("Chest Pain Type", ['TA', 'ATA', 'NAP', 'ASY'])
resting_blood_pressure = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
serum_cholesterol = st.slider("Serum Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ['Yes', 'No'])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
max_heart_rate = st.slider("Max Heart Rate Achieved", 60, 220, 150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina", ['Yes', 'No'])
st_depression = st.slider("ST Depression", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", ['Upsloping', 'Flat', 'Downsloping'])

if st.button("Predict"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_blood_pressure,
        'Cholesterol': serum_cholesterol,
        'MaxHR': max_heart_rate,
        'ST_Depression': st_depression,
        'Sex_' + sex: 1,
        'ChestPainType_'+chest_pain_type:1,
        'RestingECG_'+ resting_ecg:1,
        'ExerciseAngina_'+ exercise_induced_angina:1,
        'slope'+ slope:1
    }
    input_data = pd.DataFrame([raw_input], columns=expected_colums).fillna(0)
    
    for col in expected_colums:
        if col not in input_data.columns:
            input_data[col] = 0
            
    input_data = input_data[expected_colums]
    
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]
    
    if prediction == 1:
        st.error("High likelihood of heart disease. Please consult a doctor.")
    else:
        st.success("Low likelihood of heart disease. Keep up the healthy lifestyle!")
        