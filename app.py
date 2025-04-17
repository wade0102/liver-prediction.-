
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="نظام التنبؤ بتليف الكبد", layout="centered")

# تحميل النموذج
model = joblib.load('rf_model.pkl')

st.title("نظام التنبؤ بتليف الكبد باستخدام تقنية الغابة العشوائية")

st.markdown("يرجى إدخال بيانات المريض بدقة:")

age = st.number_input("العمر (بالسنوات)", min_value=0)
bilirubin = st.number_input("مستوى البيليروبين الكلي (ملغم/ديسيلتر)")
alk_phosphate = st.number_input("مستوى الفوسفاتاز القلوي")
sgpt = st.number_input("SGPT (ALT)")
sgot = st.number_input("SGOT (AST)")
total_protein = st.number_input("إجمالي البروتين (جم/ديسيلتر)")

if st.button("عرض التنبؤ"):
    input_data = pd.DataFrame([[age, bilirubin, alk_phosphate, sgpt, sgot, total_protein]],
                              columns=["Age", "Total_Bilirubin", "Alkphos", "Sgpt", "Sgot", "Total_Proteins"])
    prediction = model.predict(input_data)[0]
    st.success(f"درجة التليف المتوقعة: {prediction}")
