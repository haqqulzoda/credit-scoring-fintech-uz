import streamlit as st
import pandas as pd
import joblib 
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Page Config
st.set_page_config(page_title='Fintech Credit Scorer', page_icon='🏦')

# 2. Load the Artifacts (Cached for Speed)
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/xgb_model.joblib')
    preprocessor = joblib.load('models/preprocessor.joblib')
    return model, preprocessor

try:
    model,preprocessor = load_artifacts()
    st.success("Models loaded successfully!")
except FileNotFoundError: 
    st.error("Models not found. Please check the model path.")
    st.stop()

# 3. Sidebar: Input Form
st.sidebar.header("User Input Features")

def user_input_features():
    #Numerical Inputs
    checking_account = st.sidebar.number_input("Checking Account (DM)", value=0)
    duration = st.sidebar.number_input("Duration (Months)", 4,72,24)
    credit_amount = st.sidebar.number_input("Credit Amount (DM)", 250, 20000, 5000)
    age = st.sidebar.slider("Age", 18, 75, 30)
    
    # Categorical Inputs 
    sex = st.sidebar.selectbox("Sex", ["male", "female"])
    housing = st.sidebar.selectbox("Housing", ["own", "rent", "free"])
   
    data = {
        'checking_status': 'no checking', 
        'duration': duration,
        'credit_history': 'existing paid',
        'purpose': 'radio/tv',
        'credit_amount': credit_amount,
        'savings_status': '<100',
        'employment': '1<=X<4',
        'installment_commitment': 4,
        'personal_status': 'male single', 
        'other_parties': 'none',
        'residence_since': 4,
        'property_magnitude': 'real estate',
        'age': age,
        'other_payment_plans': 'none',
        'housing': housing,
        'existing_credits': 1,
        'job': 'skilled',
        'num_dependents': 1,
        'own_telephone': 'yes',
        'foreign_worker': 'yes'
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# 4. Main Page
st.title("🏦 Transparent Credit Scoring API")
st.markdown("Enter applicant details on the left to predict credit risk.")

if st.button("Predict Risk"):
    try:
    # Transform the input using the saved preprocessor
        input_processed = preprocessor.transform(input_df)

    #Predict
        prediction = model.predict(input_processed)
        probability = model.predict_proba(input_processed)[0][1] # probability of Bad 

    # Display Result
        if prediction[0] == 1:
            st.error(f"⚠️ HIGH RISK (Default Probability: {probability:.1%})")
            st.markdown("**Recommendation:** Reject Loan Application")
        else:
            st.success(f"✅ LOW RISK (Default Probability: {probability:.1%})")
            st.markdown("**Recommendation:** Approve Loan Application")   
    # Optional: Add SHAP plot here if you want to get fancy (requires complex setup)
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Ensure categorical inputs match the training data exactly.")

# Show raw data for debugging
if st.checkbox("Show raw input data"):
    st.write(input_df)