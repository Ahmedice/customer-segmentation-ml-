import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ✅ تحميل النموذج، السكالير، الأعمدة المستخدمة
with open('rfm_classifier.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']
    features = data['features']  # ['recency_log', 'frequancy_log', 'monetary_log']

# الأعمدة الخام الأصلية بدون اللوغ
raw_features = ['recency', 'frequancy', 'monetary']

# خريطة أسماء المجموعات للتوضيح
cluster_names = {
    'New Comers': 'New Customers',
    'Loyal Customers': 'Loyal',
    'At Risk': 'At Risk',
    'Lost': 'Lost'
}

# واجهة التطبيق
st.title("🚀 تصنيف العملاء باستخدام RFM و RandomForest")

option = st.radio("اختر طريقة الإدخال:", ["عميل واحد يدوي", "ملف بيانات جماعي"])

# ------------------------------
# 🔹 إدخال بيانات عميل يدوي
# ------------------------------
if option == "عميل واحد يدوي":
    st.subheader("أدخل بيانات العميل (تأكد من الدقة):")

    recency = st.number_input("Recency - عدد الأيام منذ آخر تفاعل", min_value=0)
    frequency = st.number_input("Frequency - عدد مرات الشراء", min_value=0)
    monetary = st.number_input("Monetary - إجمالي القيمة المالية", min_value=0.0)

    if st.button("🔍 صنف العميل"):
        row = [recency, frequency, monetary]

        # 🔥 تحويل اللوغ بنفس أسلوب التدريب
        row_log = np.log1p(row)

        # تحويل إلى DataFrame بالأعمدة الصحيحة
        row_df = pd.DataFrame([row_log], columns=features)

        # تطبيق السكالير
        row_scaled = scaler.transform(row_df)

        # التنبؤ بالتصنيف
        prediction = model.predict(row_scaled)[0]
        group_name = cluster_names.get(prediction, "غير معروف")

        st.success(f"✅ العميل ينتمي إلى المجموعة: {group_name} ({prediction})")


# ------------------------------
# 🔹 رفع ملف بيانات جماعي
# ------------------------------
elif option == "ملف بيانات جماعي":
    st.subheader("📂 ارفع ملف بيانات العملاء (CSV أو Excel)")
    st.info(f"يجب أن يحتوي الملف على الأعمدة التالية: {raw_features}")

    uploaded_file = st.file_uploader("اختر الملف", type=["csv", "xlsx"])

    if uploaded_file:
        # قراءة الملف
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df.columns = df.columns.str.strip().str.lower()  # تنظيف أسماء الأعمدة

        st.write("📝 عرض البيانات الأولية:")
        st.dataframe(df.head())

        # التأكد من الأعمدة المطلوبة
        required_cols = set([col.lower() for col in raw_features])
        if required_cols.issubset(df.columns):

            rfm_data = df[raw_features].copy()

            # 🔥 تحويل اللوغ
            rfm_data_log = np.log1p(rfm_data)

            # إعادة تسمية الأعمدة لتطابق ما تم التدريب عليه
            rfm_data_log.columns = features

            # تطبيق السكالينغ
            rfm_data_scaled = scaler.transform(rfm_data_log)

            # التوقع
            predictions = model.predict(rfm_data_scaled)

            df['prediction'] = predictions
            df['group_name'] = df['prediction'].map(lambda x: cluster_names.get(x, "غير معروف"))

            st.success("✅ تم تصنيف جميع العملاء بنجاح")
            st.dataframe(df[['customerid'] + raw_features + ['group_name']].head())

            # رسم التوزيع
            st.subheader("📊 توزيع العملاء حسب المجموعات:")
            st.bar_chart(df['group_name'].value_counts())

            # زر تحميل النتائج
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="💾 تحميل النتائج كملف CSV",
                data=csv,
                file_name='classified_customers.csv',
                mime='text/csv'
            )

        else:
            st.error(f"⚠️ الملف يجب أن يحتوي على الأعمدة التالية: {raw_features}")

