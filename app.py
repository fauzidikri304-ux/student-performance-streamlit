import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Student Performance Prediction",
    layout="centered"
)

st.title("📊 Prediksi Performa Mahasiswa (%)")
st.write("Aplikasi Machine Learning menggunakan **Linear Regression**")

st.markdown("""
### 🔍 Metodologi
1. Upload dataset mahasiswa  
2. Encoding data kategorikal  
3. Split data training & testing  
4. Training model Linear Regression  
5. Evaluasi dan prediksi performa (%)  
""")

# ================= UPLOAD DATA =================
file = st.file_uploader(
    "📂 Upload file Student_Performance.csv",
    type=["csv"]
)

if file is not None:
    df = pd.read_csv(file)

    st.subheader("📄 Preview Data")
    st.dataframe(df.head())

    # ================= PREPROCESSING =================
    if "Extracurricular Activities" in df.columns:
        df["Extracurricular Activities"] = df["Extracurricular Activities"].map({
            "Yes": 1,
            "No": 0
        })

    X = df.drop("Performance Index", axis=1)
    y = df["Performance Index"]

    # ================= SPLIT DATA =================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ================= TRAIN MODEL =================
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ================= EVALUATION =================
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.subheader("📈 Evaluasi Model")
    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2:.2f}")
    col2.metric("MSE", f"{mse:.2f}")

    # ================= PREDICTION (%) =================
    avg_pred = np.mean(y_pred)
    performance_percent = np.clip(avg_pred, 0, 100)

    st.subheader("🎯 Prediksi Performa Mahasiswa")
    st.metric(
        label="Tingkat Performa",
        value=f"{performance_percent:.2f} %"
    )

    # ================= CATEGORY =================
    if performance_percent >= 85:
        kategori = "Sangat Baik"
    elif performance_percent >= 70:
        kategori = "Baik"
    elif performance_percent >= 55:
        kategori = "Cukup"
    else:
        kategori = "Perlu Peningkatan"

    st.success(f"📌 Kategori Performa: **{kategori}**")

else:
    st.info("Silakan upload file CSV terlebih dahulu.")
