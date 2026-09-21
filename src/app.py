import streamlit as st
import pandas as pd
import joblib


st.set_page_config(page_title="Churn Dashboard", 
                   page_icon=":bar_chart:",  
                   layout="wide")  


st.title("Churn Prediction Dashboard")


with st.expander("About model"):
    st.header("About the churn prediction model")
    st.write("The model takes 20 input features relating to a Telco subscription, including a customerID. "  
         "The types of features vary, including both categorical and numerical data. This is then run through" 
         "a logistic regression to predict whether or not the customer will churn (cancel their subscription)")




def get_user_input():
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.sidebar.selectbox('Senior Citizen', [0, 1])
    partner = st.sidebar.selectbox('Has Partner', ['No', 'Yes'])
    dependents = st.sidebar.selectbox('Has Dependents', ['No', 'Yes'])

# Account info
    tenure = st.sidebar.number_input('Tenure (months)', min_value=0, max_value=100, value=12)
    contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ['No', 'Yes'])
    payment_method = st.sidebar.selectbox('Payment Method',
    ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

# Services
    phone_service = st.sidebar.selectbox('Phone Service', ['No', 'Yes'])
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    internet_service = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.sidebar.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    online_backup = st.sidebar.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    device_protection = st.sidebar.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    tech_support = st.sidebar.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.sidebar.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])

# Charges
    monthly_charges = st.sidebar.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.sidebar.number_input('Total Charges ($)', min_value=0.0, max_value=10000.0, value=840.0)

    return {
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges}

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()

input_df = pd.DataFrame([get_user_input()])


if st.sidebar.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f'This customer is likely to churn with {probability:.1%} probability')

    else:
        st.success(f'This customer is likely to stay, but has a {probability:.1%} probability of churn')

else: 
    st.info('Adjust the features on the sidebar, then press predict')





