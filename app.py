import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page title
st.set_page_config(page_title="Telco Customer Churn Predictor", layout="centered")

# Load pre-trained model and preprocessor
@st.cache_resource
def load_assets():
    model = joblib.load("churn_model.joblib")
    preprocessor = joblib.load("preprocessor.joblib")
    return model, preprocessor

try:
    model, preprocessor = load_assets()
except Exception as e:
    st.error(f"Error loading model or preprocessor files. Make sure 'churn_model.joblib' and 'preprocessor.joblib' are in the same folder. Details: {e}")
    st.stop()

st.title("📞 Telco Customer Churn Predictor")
st.write("Enter customer details below to predict the probability of churn.")

# Create Form Inputs organized by categories
with st.form("churn_form"):
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Is Senior Citizen?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col2:
        Partner = st.selectbox("Has Partner?", ["Yes", "No"])
        Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

    st.subheader("Services Used")
    col3, col4, col5 = st.columns(3)
    with col3:
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    with col4:
        OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    with col5:
        TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.subheader("Billing & Contract Details")
    col6, col7 = st.columns(2)
    with col6:
        Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    with col7:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)

    submit_btn = st.form_submit_button("Predict Churn Risk")

if submit_btn:
    # Build input DataFrame
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }])

    # Predict
    prob = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.write("--- ")
    if prediction == 1:
        st.error(f"⚠️ **High Risk!** This customer is predicted to **CHURN** with a probability of **{prob:.1%}**.")
    else:
        st.success(f"✅ **Safe!** This customer is predicted to **STAY** (Churn probability: **{prob:.1%}**).")
