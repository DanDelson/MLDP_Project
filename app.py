import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load Model, Model is trainned by Spambase dataset from UCI ML Repo, and uses numeric features such as frequency of certain words and average capital run length to predict if an email is spam or not
model = joblib.load('spambase_best_rs_rf_model.pkl')

## Streamlit App
st.title("Spam Email Detection")

## User Input, Sliders are use to determine how frquent a word appears in the email, and the average capital run is also a feature used in the model
freeFreq = st.slider("Frequency of 'free'", 0.0, 100.0)
moneyFreq = st.slider("Frequency of 'money'", 0.0, 100.0)
dollarFreq = st.slider("Frequency of '$'", 0.0, 100.0)
capitalFreq = st.slider("Average Capital Run", 0.0, 10.0)


if st.button("Predict"):
    prediction = model.predict([[freeFreq, moneyFreq, dollarFreq, capitalFreq]])
    if prediction[0] == 1:
        st.success("The email is classified as SPAM.")
    else:
        st.success("The email is classified as HAM.")