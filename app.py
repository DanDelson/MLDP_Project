import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load Model
model = joblib.load('spambase_best_rs_rf_model.pkl')

## Streamlit App
st.title("Spam Email Detection")

## User Input
user_input = st.text_area ("Enter email text")

if st.button("Predict"):
    prediction = model.predict(processed_text)

