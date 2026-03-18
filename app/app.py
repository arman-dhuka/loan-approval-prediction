import streamlit as st
import pandas as pd
import joblib

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# -------------------------
# CUSTOM CSS (UI MAGIC 🔥)
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1 {
    color: #2C3E50;
    text-align: center;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stButton>button:hover {
    background-color: #45a049;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("models/loan_model.pkl")

# -------------------------
# TITLE
# -------------------------
st.markdown("<h1>🏦 Loan Approval Prediction</h1>", unsafe_allow_html=True)
st.write("### 💼 Fill applicant details to check loan approval")

# -------------------------
# LAYOUT (2 COLUMN DESIGN 🔥)
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Personal Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    st.markdown("### 💰 Financial Info")
    app_income = st.number_input("Applicant Income", min_value=0)
    co_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Amount Term", min_value=0)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# -------------------------
# PREPROCESS FUNCTION
# -------------------------
def preprocess_input():
    dep = 3 if dependents == "3+" else int(dependents)

    data = {
        "Gender": 1 if gender == "Male" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Dependents": dep,
        "Education": 1 if education == "Graduate" else 0,
        "Self_Employed": 1 if self_employed == "Yes" else 0,
        "ApplicantIncome": app_income,
        "CoapplicantIncome": co_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0
    }

    return pd.DataFrame([data])

# -------------------------
# PREDICT BUTTON
# -------------------------
st.markdown("---")

if st.button("🔍 Predict Loan Status"):
    input_df = preprocess_input()

    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]

    st.markdown("### 📊 Prediction Result")

    if prediction[0] == 1:
        st.success(f"✅ Loan Approved (Confidence: {prob:.2f})")
    else:
        st.error(f"❌ Loan Rejected (Confidence: {1-prob:.2f})")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown(
    "<center>Made with ❤️ by Arman Dhuka | ML Project</center>",
    unsafe_allow_html=True
)