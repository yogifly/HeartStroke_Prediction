import streamlit as st
import pandas as pd
import joblib

# ======================
# Load Saved Model & Scaler
# ======================
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")

# ======================
# Streamlit Page Config
# ======================
st.set_page_config(page_title="Stroke Risk Predictor", layout="wide", page_icon="🧠")

# ======================
# Custom Styling (black text in sidebar)
# ======================
st.markdown("""
    <style>
        .main {
            background-color: #f9fafc;
            padding: 2rem;
            border-radius: 10px;
        }
        h1 {
            text-align: center;
            color: #1e3a8a;
        }
        .stButton>button {
            background-color: #2563eb;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #1e40af;
        }
        .sidebar-content {
            background-color: #eef2ff;
            padding: 1rem;
            border-radius: 10px;
        }
        .sidebar-info {
            background-color: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            color: black;  /* 🔹 Ensures black text inside sidebar */
        }
        .sidebar-info h4 {
            color: #1e3a8a;
            margin-bottom: 0.5rem;
        }
        .sidebar-info p {
            margin: 0.2rem 0;
            font-size: 0.9rem;
            color: black; /* 🔹 Black text for paragraphs */
        }
        .sidebar small {
            color: black !important; /* 🔹 Even footer note text black */
        }
    </style>
""", unsafe_allow_html=True)

# ======================
# Sidebar with Average Values
# ======================
st.sidebar.title("📋 Reference Health Indicators")

st.sidebar.markdown("""
Below are **average values**  
to guide your input selections:
""")

st.sidebar.markdown("""
<div class="sidebar-info">
    <h4>🧑 Demographics</h4>
    <p><b>Average Age:</b> 45 years</p>
    <p><b>Married Population:</b> ~70%</p>
    <p><b>Urban Residents:</b> ~55%</p>
</div>

<div class="sidebar-info">
    <h4>🩺 Health Parameters</h4>
    <p><b>Avg Glucose Level:</b> 100–120 mg/dL</p>
    <p><b>Average BMI:</b> 24–28</p>
    <p><b>Hypertension Prevalence:</b> ~13%</p>
    <p><b>Heart Disease Prevalence:</b> ~5%</p>
</div>

<div class="sidebar-info">
    <h4>🚬 Lifestyle</h4>
    <p><b>Never Smoked:</b> ~55%</p>
    <p><b>Former Smokers:</b> ~25%</p>
    <p><b>Current Smokers:</b> ~15%</p>
</div>

<hr style="margin-top:1rem; margin-bottom:0.5rem;">
""", unsafe_allow_html=True)
st.sidebar.markdown("""
---
💡 **Tip:**  
Higher glucose or BMI values increase stroke risk.  
Use realistic values for accurate prediction.
""")

# ======================
# Main UI
# ======================
st.title("🧠 Stroke Risk Prediction")
st.markdown("Enter the patient details below to assess **stroke risk** using the trained ML model.")

# Input Form
with st.form("stroke_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age (in years)", min_value=1, max_value=120, value=45)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])

    with col2:
        work_type = st.selectbox("Work Type", ["children", "Govt_job", "Never_worked", "Private", "Self-employed"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=40.0, max_value=300.0, value=110.0, format="%.1f")
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0, format="%.1f")
        smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    submitted = st.form_submit_button("🔍 Predict Stroke Risk")

# ======================
# Encoding & Prediction
# ======================
if submitted:
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

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("🩺 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Stroke Detected! (Probability: **{prob:.2f}**) ")
    else:
        st.success(f"✅ Low Risk of Stroke (Probability: **{prob:.2f}**)")

