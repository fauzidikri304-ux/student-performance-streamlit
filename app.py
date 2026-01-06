import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Performance Prediction")
st.title("🎓 Prediksi Performa Mahasiswa (%)")

file = st.file_uploader("📂 Upload Student_Performance.csv", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    # Encoding
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map({
        "Yes": 1,
        "No": 0
    })

    X = df.drop("Performance Index", axis=1)
    y = df["Performance Index"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluasi
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.subheader("📊 Evaluasi Model")
    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2:.2f}")
    col2.metric("MSE", f"{mse:.2f}")

    # ================= INPUT MANUAL =================
    st.subheader("🧮 Input Data Mahasiswa")

    hours = st.number_input("Jam Belajar", 0, 24, 5)
    prev = st.number_input("Nilai Sebelumnya", 0, 100, 70)
    extra = st.selectbox("Ekstrakurikuler", ["Yes", "No"])
    sleep = st.number_input("Jam Tidur", 0, 24, 7)
    papers = st.number_input("Latihan Soal", 0, 10, 2)

    extra_val = 1 if extra == "Yes" else 0

    if st.button("🔮 Prediksi"):
        input_df = pd.DataFrame([{
            "Hours Studied": hours,
            "Previous Scores": prev,
            "Extracurricular Activities": extra_val,
            "Sleep Hours": sleep,
            "Sample Question Papers Practiced": papers
        }])

        pred = model.predict(input_df)[0]
        pred_percent = np.clip(pred, 0, 100)

        # Kategori
        if pred_percent >= 85:
            kategori = "Sangat Baik"
        elif pred_percent >= 70:
            kategori = "Baik"
        elif pred_percent >= 55:
            kategori = "Cukup"
        else:
            kategori = "Perlu Peningkatan"

        st.success(f"🎯 Prediksi Performa: **{pred_percent:.2f}%**")
        st.info(f"📌 Kategori: **{kategori}**")

    # ================= GRAFIK =================
    st.subheader("📈 Jam Belajar vs Performance Index")
    fig, ax = plt.subplots()
    ax.scatter(df["Hours Studied"], y)
    ax.set_xlabel("Jam Belajar")
    ax.set_ylabel("Performance Index")
    st.pyplot(fig)

else:
    st.info("Silakan upload file CSV terlebih dahulu.")
