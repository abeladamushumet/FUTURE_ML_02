import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===========================
#Load Trained Model
# ===========================
with open("D:/Projects/FUTURE INTERN PROJECTS/FUTURE_ML_02/model/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# ===========================
# Load Training Features
# ===========================
X_train = pd.read_csv("D:/Projects/FUTURE INTERN PROJECTS/FUTURE_ML_02/data/processed/x_train.csv")
model_features = [str(col) for col in X_train.columns]  # ensure all strings

# ===========================
# Streamlit App Title
# ===========================
st.title("Customer Churn Prediction System")
st.write("Enter customer info to predict churn probability.")

# ===========================
# Sidebar Inputs (example for common features)
# ===========================
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=50.0)
TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0, value=500.0)

Contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

# ===========================
#  Prepare Input Data
# ===========================
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "Contract_Month-to-month": 1 if Contract=="Month-to-month" else 0,
    "Contract_One year": 1 if Contract=="One year" else 0,
    "Contract_Two year": 1 if Contract=="Two year" else 0,
    "PaymentMethod_Bank transfer": 1 if PaymentMethod=="Bank transfer" else 0,
    "PaymentMethod_Credit card": 1 if PaymentMethod=="Credit card" else 0,
    "PaymentMethod_Electronic check": 1 if PaymentMethod=="Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if PaymentMethod=="Mailed check" else 0
}

input_df = pd.DataFrame([input_dict])

# ===========================
# Align Input with Model Features
# ===========================
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0  # missing columns = 0

input_df = input_df[model_features]  # reorder columns
input_df.columns = input_df.columns.astype(str)  # ensure all string names

# ===========================
#  Predict Churn
# ===========================
prediction = model.predict(input_df)[0]
prediction_prob = model.predict_proba(input_df)[0][1]

st.subheader("Prediction Results")
st.write("**Churn Prediction:**", "Yes" if prediction==1 else "No")
st.write("**Probability of Churn:**", f"{prediction_prob:.2f}")

# ===========================
#  Feature Importance 
# ===========================
if hasattr(model, "feature_importances_"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    feat_imp = pd.Series(model.feature_importances_, index=model_features).sort_values(ascending=False)[:20]
    st.subheader("Top Feature Importances")
    st.bar_chart(feat_imp)
