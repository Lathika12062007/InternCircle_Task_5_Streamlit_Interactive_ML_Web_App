
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("titanic_survival_model.pkl")

# Page settings
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Title
st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter passenger details to predict Titanic survival."
)

# Inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Gender",
    ["female", "male"]
)

age = st.number_input(
    "Age",
    min_value=1.0,
    max_value=100.0,
    value=30.0
)

sibsp = st.number_input(
    "Siblings / Spouses",
    min_value=0,
    max_value=8,
    value=0,
    step=1
)

parch = st.number_input(
    "Parents / Children",
    min_value=0,
    max_value=6,
    value=0,
    step=1
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    max_value=600.0,
    value=32.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"]
)

# Prediction
if st.button("Predict Survival"):

    input_data = pd.DataFrame({
        "pclass": [pclass],
        "sex": [sex],
        "age": [age],
        "sibsp": [sibsp],
        "parch": [parch],
        "fare": [fare],
        "embarked": [embarked]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Passenger is predicted to survive.")
    else:
        st.error("❌ Passenger is predicted not to survive.")

    st.metric(
        "Survival Probability",
        f"{probability * 100:.2f}%"
    )

    st.subheader("Passenger Details")
    st.dataframe(input_data)
