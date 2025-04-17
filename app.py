import streamlit as st
import joblib
import numpy as np

# تحميل النموذج
model = joblib.load('rf_model.pkl')

st.title("توقع مرض الكبد")

# مدخلات المستخدم
age = st.number_input("العمر", 1, 100, 30)
gender = st.selectbox("الجنس", ["ذكر", "أنثى"])
total_bilirubin = st.number_input("Total Bilirubin", 0.0, 10.0, 1.0)
alkaline_phosphotase = st.number_input("Alkaline Phosphotase", 0, 2000, 200)
alamine_aminotransferase = st.number_input("Alamine Aminotransferase", 0, 2000, 50)
aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", 0, 2000, 50)
total_proteins = st.number_input("Total Proteins", 0.0, 10.0, 6.5)
albumin = st.number_input("Albumin", 0.0, 6.0, 3.5)
ag_ratio = st.number_input("Albumin and Globulin Ratio", 0.0, 5.0, 1.0)

if st.button("توقّع"):
    gender_encoded = 1 if gender == "ذكر" else 0
    input_data = np.array([[age, gender_encoded, total_bilirubin,
                            alkaline_phosphotase, alamine_aminotransferase,
                            aspartate_aminotransferase, total_proteins,
                            albumin, ag_ratio]])
    
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("المريض يعاني من مشاكل في الكبد")
    else:
        st.success("المريض لا يعاني من مشاكل في الكبد")