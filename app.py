import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load Model, Model is trainned by Spambase dataset from UCI ML Repo, and uses numeric features such as frequency of certain words and average capital run length to predict if an email is spam or not
model = joblib.load('spambase_best_rs_rf_model.pkl')
# Load Models Columns name
feature_cols = joblib.load("feature_columns.pkl")

example_spam = joblib.load("example_spam.pkl")
example_ham = joblib.load("example_ham.pkl")

## Streamlit App
st.title("Spam Email Detection")

st.markdown("""
    This application predicts whether an email is spam or not based on various features
    extracted from the email content. Please input the feauture value frequency below to get the prediction.
""")

if st.button("Load Example Spam"):
    st.session_state.inputs = example_spam.copy()

if st.button("Load Example Ham"):
    st.session_state.inputs = example_ham.copy()

## Check if list has input saved in session state, if no, create placeholder 0.0 value for each features column
if "inputs" not in st.session_state:
    st.session_state.inputs = [0.0] * len(feature_cols)

user_inputs = []

st.subheader("Enter Feature Values")

with st.expander("Feature Descriptions"):
    st.info("Enter feature values based on email statistics. Default values are zero")
    for i, col in enumerate(feature_cols):
        if "capital" in col:
            value = st.number_input(col, min_value=0.0, value=st.session_state.inputs[i])
        else:
            value = st.slider(col, 0.0, 100.0, st.session_state.inputs[i])

        user_inputs.append(value)

if st.button("Predict"):
    if sum(user_inputs) == 0:
        st.warning("Please enter at least one non-zero value to make a prediction.")
        st.stop()

    prediction = model.predict([user_inputs])[0]
    prob = model.predict_proba([user_inputs])[0][1]

    st.divider()
    st.subheader("Prediction Result")
    st.progress(float(prob))
    
    if prediction == 1:
        st.error(f"SPAM detected (Probability: {prob:.2%})")
    else:
        st.success(f"Not spam (Probability: {prob:.2%})")