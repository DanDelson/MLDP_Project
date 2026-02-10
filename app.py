import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load Model, Model is trainned by Spambase dataset from UCI ML Repo, and uses numeric features such as frequency of certain words and average capital run length to predict if an email is spam or not
model = joblib.load('spambase_best_rs_rf_model.pkl')
# Load Models Columns name
feature_cols = joblib.load("feature_columns.pkl")

## Streamlit App
st.title("Spam Email Detection")

user_inputs = []

st.subheader("Enter Feature Values")

with st.expander("Feature Descriptions"):
    for col in feature_cols:
        value = st.number_input(col, min_value=0.0, value=0.0)
        user_inputs.append(value)

if st.button("Predict"):

    if sum(user_inputs) == 0:
        st.warning("Please enter at least one non-zero value to make a prediction.")
        st.stop()

    prediction = model.predict([user_inputs])[0]
    prob = model.predict_proba([user_inputs])[0][1]

    if prediction == 1:
        st.error(f"SPAM detected (Probability: {prob:.2%})")
    else:
        st.success(f"Not spam (Probability: {prob:.2%})")