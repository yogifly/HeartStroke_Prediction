import streamlit as st
import pandas as pd
import joblib

# ======================
# Load Saved Model & Scaler
# ======================
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")   # save your scaler too during training

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="Stroke Prediction", layout="centered")

st.title("🩺 Stroke Prediction App")
st.write("Enter patient details below to predict stroke risk:")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 1, 100, 50)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["children", "Govt_job", "Never_worked", "Private", "Self-employed"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=40.0, max_value=300.0, value=120.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# ======================
# Encode Inputs (same as training)
# ======================
input_dict = {
    "gender": 1 if gender == "Male" else (2 if gender == "Other" else 0),
    "age": age,
    "hypertension": 1 if hypertension == "Yes" else 0,
    "heart_disease": 1 if heart_disease == "Yes" else 0,
    "ever_married": 1 if ever_married == "Yes" else 0,
    "work_type": {"children": 0, "Govt_job": 1, "Never_worked": 2, "Private": 3, "Self-employed": 4}[work_type],
    "Residence_type": 1 if residence_type == "Urban" else 0,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": {"formerly smoked": 0, "never smoked": 1, "smokes": 2, "Unknown": 3}[smoking_status]
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Scale numeric features
input_scaled = scaler.transform(input_df)

# ======================
# Prediction
# ======================
if st.button("🔍 Predict Stroke Risk"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Stroke!")
    else:
        st.success(f"✅ Low Risk of Stroke ")
