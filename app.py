# =========================
# app.py - Streamlit Loan Prediction App
# =========================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import random

# =========================
# Load model & results
# =========================
model = joblib.load("best_model.joblib")        # your trained best model
results_df = pd.read_csv("model_results.csv")   # comparison of models

st.set_page_config(page_title="🏦 Loan Prediction App", layout="wide")

st.title("🏦 Loan Approval Prediction")
st.write("This app predicts whether a loan application will be **Approved** or **Rejected** based on applicant details.")


# =========================
# Show Model Comparison
# =========================
st.subheader("📊 Model Comparison")
st.dataframe(results_df)

def fill_random_inputs():
    return {
        "Gender": random.choice(["Male", "Female"]),
        "Married": random.choice(["Yes", "No"]),
        "Dependents": random.choice(["0", "1", "2", "3+"]),
        "Education": random.choice(["Graduate", "Not Graduate"]),
        "Self_Employed": random.choice(["Yes", "No"]),
        "ApplicantIncome": random.randint(1000, 25000),
        "CoapplicantIncome": random.randint(0, 15000),
        "LoanAmount": random.randint(50, 700),
        "Loan_Amount_Term": random.choice([180, 360, 480, 600]),
        "Credit_History": random.choice([1.0, 0.0]),
        "Property_Area": random.choice(["Urban", "Rural", "Semiurban"])
    }

# =========================
# Session state for input
# =========================
if "input_data" not in st.session_state:
    st.session_state.input_data = fill_random_inputs()

# =========================
# Input fields in one box side by side
# =========================
with st.expander("Loan Applicant Details", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.input_data["Gender"] = st.selectbox("Gender", ["Male","Female"], index=["Male","Female"].index(st.session_state.input_data["Gender"]))
        st.session_state.input_data["Married"] = st.selectbox("Married", ["Yes","No"], index=["Yes","No"].index(st.session_state.input_data["Married"]))
        st.session_state.input_data["Dependents"] = st.selectbox("Dependents", ["0","1","2","3+"], index=["0","1","2","3+"].index(st.session_state.input_data["Dependents"]))
        st.session_state.input_data["Education"] = st.selectbox("Education", ["Graduate","Not Graduate"], index=["Graduate","Not Graduate"].index(st.session_state.input_data["Education"]))

    with col2:
        st.session_state.input_data["Self_Employed"] = st.selectbox("Self Employed", ["Yes","No"], index=["Yes","No"].index(st.session_state.input_data["Self_Employed"]))
        st.session_state.input_data["ApplicantIncome"] = st.number_input("Applicant Income", min_value=0, step=500, value=st.session_state.input_data["ApplicantIncome"])
        st.session_state.input_data["CoapplicantIncome"] = st.number_input("Coapplicant Income", min_value=0, step=500, value=st.session_state.input_data["CoapplicantIncome"])
        st.session_state.input_data["LoanAmount"] = st.number_input("Loan Amount", min_value=0, step=10, value=st.session_state.input_data["LoanAmount"])

    with col3:
        st.session_state.input_data["Loan_Amount_Term"] = st.number_input("Loan Term (days)", min_value=0, step=30, value=st.session_state.input_data["Loan_Amount_Term"])
        st.session_state.input_data["Credit_History"] = st.selectbox("Credit History", [1.0, 0.0], index=[1.0,0.0].index(st.session_state.input_data["Credit_History"]))
        st.session_state.input_data["Property_Area"] = st.selectbox("Property Area", ["Urban","Rural","Semiurban"], index=["Urban","Rural","Semiurban"].index(st.session_state.input_data["Property_Area"]))

# =========================
# Buttons
# =========================

    if st.button("🎲 Fill Random Inputs"):
        st.session_state.input_data = fill_random_inputs()
        st.rerun()


    if st.button("🚀 Predict Loan Approval"):
        input_df = pd.DataFrame([st.session_state.input_data])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        if prediction == 1:
            st.success(f"✅ Loan Approved with probability {proba:.2%}")
            st.progress(int(proba * 100))
        else:
            st.error(f"❌ Loan Rejected with probability {1-proba:.2%}")
            st.progress(int((1 - proba) * 100))


# Bigger graph
# Put the chart inside a narrower column to reduce its size
col1, col2, col3 = st.columns([1,2,1])   # middle column narrower
with col2:
    fig, ax = plt.subplots(figsize=(4, 2.5))  # smaller figure
    results_df.plot(
        x="Model", y="F1", kind="barh",
        ax=ax, legend=False, color="skyblue"
    )
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Performance (F1 Score)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Models")
    st.pyplot(fig)



# =========================
# Project Summary
# =========================
st.subheader("📖 Project Summary")
st.markdown(
    """
    ### 🔹 Steps we followed in this project:
    1. **Data Preprocessing**  
       - Handled missing values (LoanAmount, Credit_History, etc.)  
       - Encoded categorical variables (Gender, Education, Property Area)  
       - Scaled numerical features for algorithms like SVC & KNN  

    2. **Pipelines**  
       - Used `Pipeline` & `ColumnTransformer` to automate preprocessing  
       - Ensured consistent transformation for both training & prediction  

    3. **Model Training & Tuning**  
       - Trained multiple ML models (Logistic Regression, Random Forest, Gradient Boosting, SVC, Decision Tree, KNN)  
       - Used `RandomizedSearchCV` for hyperparameter tuning  
       - Compared models using **Accuracy, Precision, Recall, F1-score**  

    4. **Model Selection**  
       - Saved the best-performing model as `best_model.joblib`  
       - Created a CSV (`model_results.csv`) to store all model performances  

    5. **Deployment**  
       - Built this **Streamlit App** to make predictions in real-time
    """
)

# =========================
# Real-world Applications
# =========================
st.subheader("🌍 Real-World Applications")
st.markdown(
    """
    - **Banks & Financial Institutions** → Automating loan approval decisions  
    - **Credit Risk Analysis** → Identifying high-risk applicants  
    - **FinTech Startups** → Real-time loan eligibility checks  
    - **Customer Experience** → Faster, transparent approval process  
    - **Fraud Detection** → Combining with anomaly detection for suspicious applications  
    ## Made with ❤️ by Kanav Chauhan
    """
)

# =========================
# Refresh Button
# =========================
if st.button("🔄 Reset App"):
    st.rerun()
