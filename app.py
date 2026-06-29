import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(
    page_title="Financial Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "Bootcamp_project_cleaned.csv")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    return df


@st.cache_data
def encode_data(df):
    categorical_cols = ["Transaction_Type", "Device_Used", "Location", "Payment_Method"]
    encoders = {}
    df_encoded = df.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    return df_encoded, encoders


df = load_data()
df_encoded, encoders = encode_data(df)

page = st.sidebar.radio(
    "Navigation",
    ["Home", "EDA", "Models", "Prediction"]
)

# ================= HOME =================
if page == "Home":
    st.title("💳 Fraud Detection System")

    fraud = df["Fraudulent"].sum()
    legit = len(df) - fraud

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", len(df))
    col2.metric("Fraud", fraud)
    col3.metric("Legit", legit)

    st.dataframe(df.head(10))


# ================= EDA =================
elif page == "EDA":
    st.title("📊 Data Analysis")

    st.subheader("Fraud Distribution")
    fig = px.pie(df, names="Fraudulent")
    st.plotly_chart(fig)

    st.subheader("Transaction Amount")
    fig2 = px.histogram(df, x="Transaction_Amount", nbins=50)
    st.plotly_chart(fig2)

    st.subheader("Fraud by Type")
    fig3 = px.bar(df.groupby("Transaction_Type")["Fraudulent"].sum().reset_index(),
                  x="Transaction_Type", y="Fraudulent")
    st.plotly_chart(fig3)


# ================= MODELS =================
elif page == "Models":
    st.title("🤖 Machine Learning")

    X = df_encoded.drop(columns=["Fraudulent", "Transaction_ID", "User_ID"], errors="ignore")
    y = df_encoded["Fraudulent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    acc = accuracy_score(y_test, pred)

    st.success(f"Random Forest Accuracy: {acc:.2f}")

    st.text(classification_report(y_test, pred))

    joblib.dump(rf, "fraud_model.pkl")
    joblib.dump(encoders, "encoders.pkl")


# ================= PREDICTION =================
elif page == "Prediction":
    st.title("🔍 Real-Time Prediction")

    try:
        model = joblib.load("fraud_model.pkl")
        encoders = joblib.load("encoders.pkl")
    except:
        st.error("⚠️ Please train model first in Models tab")
        st.stop()

    amount = st.number_input("Transaction Amount", 0.0)
    tx_type = st.selectbox("Transaction Type", df["Transaction_Type"].unique())
    device = st.selectbox("Device", df["Device_Used"].unique())
    location = st.selectbox("Location", df["Location"].unique())
    payment = st.selectbox("Payment Method", df["Payment_Method"].unique())
    hour = st.number_input("Hour", 0, 23)

    if st.button("Predict"):
        try:
            X_input = pd.DataFrame([{
                "Transaction_Amount": amount,
                "Transaction_Type": encoders["Transaction_Type"].transform([tx_type])[0],
                "Device_Used": encoders["Device_Used"].transform([device])[0],
                "Location": encoders["Location"].transform([location])[0],
                "Payment_Method": encoders["Payment_Method"].transform([payment])[0],
                "Time_of_Transaction": hour
            }])

            pred = model.predict(X_input)[0]

            if pred == 1:
                st.error("🚨 FRAUD DETECTED")
            else:
                st.success("✅ SAFE TRANSACTION")

        except Exception as e:
            st.error(f"Error: {e}")
