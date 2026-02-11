import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# ------------------ Page Config ------------------
st.set_page_config(page_title="Insurance Prediction App", layout="centered")
st.title("Insurance Charges Prediction")

# ------------------ Load CSV Safely ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "insurance.csv")
df = pd.read_csv(csv_path)

# ------------------ Features and Target ------------------
X = df.drop(columns=["charges"])
y = df["charges"]

# ------------------ Train/Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ Preprocessing ------------------
num_features = ["age", "bmi", "children"]
cat_features = ["sex", "smoker", "region"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# ------------------ Model Pipeline ------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

# ------------------ UI Inputs ------------------
age = st.number_input("Age", min_value=18, max_value=100, value=25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northwest", "northeast"])

# ------------------ Predict ------------------
input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

prediction = model.predict(input_df)[0]

st.subheader(f"Predicted Insurance Charges: ${prediction:,.2f}")

# ------------------ Graph ------------------
if st.checkbox("Show Actual vs Predicted Charges", key="show_graph"):
    y_pred = model.predict(X_test)

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()], "r--")
    ax.set_xlabel("Actual Charges ($)")
    ax.set_ylabel("Predicted Charges ($)")
    ax.set_title("Actual vs Predicted Insurance Charges")

    st.pyplot(fig)
