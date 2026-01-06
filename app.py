import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Page config
st.set_page_config(
    page_title="Student Performance Prediction",
    layout="centered"
)

st.title("📊 Prediksi Performance Mahasiswa")
st.write("Aplikasi Machine Learning menggunakan **Linear Regression**")

# Upload file
file = st.file_uploader(
    "📂 Upload file Student_Performance.csv",
    type=["csv"]
)

if file is not None:
    df = pd.read_csv(file)

    st.subheader("📄 Preview Data")
    st.dataframe(df.head())

    # Encoding
    if "Extracurricular Activities" in df.columns:
        df["Extracurricular Activities"] = df["Extracurricular Activities"].map({
            "Yes": 1,
            "No": 0
        })

    X = df.drop("Performance Index", axis=1)
    y = df["Performance Index"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.subheader("📈 Hasil Evaluasi Model")
    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2:.2f}")
    col2.metric("MSE", f"{mse:.2f}")

    st.success("✅ Model berhasil dijalankan!")
else:
    st.info("Silakan upload file CSV terlebih dahulu.")
