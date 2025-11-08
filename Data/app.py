import streamlit as st
import pandas as pd
import joblib
import os


model = joblib.load("voting_model.pkl")
threshold = 0.062

st.title("🧠 Stroke Risk Predictor")

st.sidebar.header("Prediction Mode")
mode = st.sidebar.radio("Choose input method:", ["📁 Upload File", "📝 Manual Entry"])

# ----------------------------------------------
# File Upload Mode
# ----------------------------------------------
if mode == "📁 Upload File":
    st.write("Upload a CSV, XLS, or XLSX file, and the model will make predictions!")

    uploaded_file = st.file_uploader("Upload your file", type=["csv", "xls", "xlsx"])

    if uploaded_file:
        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):
            input_data = pd.read_csv(uploaded_file)
        elif filename.endswith(".xls"):
            input_data = pd.read_excel(uploaded_file, engine="xlrd")
        elif filename.endswith(".xlsx"):
            input_data = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            st.error("Unsupported file type.")
            st.stop()

        input_data["age_hypertension"] = input_data["age"] * input_data["hypertension"]

        Q1 = input_data["bmi"].quantile(0.25)
        Q3 = input_data["bmi"].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        input_data["bmi_outlier"] = (input_data["bmi"] > upper_bound).astype(int)

        st.write("### 🧾 Uploaded Data", input_data)

        probas = model.predict_proba(input_data)[:, 1]
        predictions = (probas >= threshold).astype(int)
        input_data["Prediction (%)"] = (probas * 100).round(2)
        input_data["Predicted Stroke Class"] = predictions

        st.write("### 📊 Predictions", input_data)

        csv = input_data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")

# ----------------------------------------------
# Manual Input Mode
# ----------------------------------------------
else:
    st.write("Enter the details below to estimate stroke risk.")

    with st.form("manual_input_form"):
        age = st.number_input("Age", min_value=1, max_value=120, value=40)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=270.0,  value=100.0)
        bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0)
        smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes"])
        gender = st.selectbox("Gender", ["Male", "Female"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        df_manual = pd.DataFrame([{
            "age": age,
            "hypertension": 1 if hypertension == "Yes" else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence_type,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "smoking_status": smoking_status,
            "gender": gender
        }])

        df_manual["age_hypertension"] = df_manual["age"] * df_manual["hypertension"]

        Q1 = bmi * 0.95
        Q3 = bmi * 1.05
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        df_manual["bmi_outlier"] = int(bmi > upper_bound)

        probas = model.predict_proba(df_manual)[:, 1]
        prediction = int(probas[0] >= threshold)

        st.markdown("### 🧾 Input Summary")
        st.write(df_manual)

        st.markdown("### 🎯 Prediction Result")
        st.write(f"**Stroke Prediction**: {'Yes' if prediction else 'No'}")
        st.write(f"**Risk Score**: {probas[0]*100:.2f}%")

