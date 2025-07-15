import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
from streamlit_shap import st_shap

# Page config
st.set_page_config(page_title="Thyroid Cancer Recurrence Prediction", page_icon="🧠", layout="wide")

# Load models and tools
model = joblib.load("stacking_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
X_train = joblib.load("X_train.pkl")

# --------------------------
# 🧠 Title
# --------------------------
st.markdown("<h2 style='text-align: center; font-size: 42px;color: #2c3e50;'>🧠 Thyroid Cancer Recurrence Prediction System</h2>", unsafe_allow_html=True)
st.markdown("---")
# System introduction
st.markdown("""
<div style='text-align: justify; color: #34495e; font-size: 20px; line-height: 1.6;'>
This is a <b>Thyroid Cancer Recurrence Prediction System</b> based on stacking model architecture. Thestacking model integrates multiple machine learning algorithms to enhance prediction accuracy. itcombines predictions from five base models (Random Forest, Gradient Boosting, XGBoost, Support VectorMachine (SVC), and K-Nearest Neighbors (KNN) and uses a logistic regression as meta-model to makefinal predictions. The system is designed to help medical professionals assess the risk of thyroid cancerrecurrence by analyzing patient-specific features. lt provides a data-driven approach to support clinicaldecision-making.<br>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --------------------------
# 📝 Input form
# --------------------------
with st.form("prediction_form"):
    st.markdown("#### 🧾 Input Patient Features")
    cols = st.columns(3)
    user_input = {}

    for idx, feature in enumerate(features):
        with cols[idx % 3]:
            if feature in label_encoders:
                options = label_encoders[feature].classes_.tolist()
                user_input[feature] = st.selectbox(f"{feature}", options)
            elif feature == "Age":
                user_input[feature] = st.number_input("Age", min_value=0, max_value=120, value=50)
            else:
                user_input[feature] = st.number_input(f"{feature}", value=0.0)

    submitted = st.form_submit_button("✅ Predict")

# --------------------------
# 🔮 Prediction & SHAP
# --------------------------
if submitted:
    input_df = pd.DataFrame([user_input])

    # Encode & scale
    for col in label_encoders:
        if col in input_df.columns:
            input_df[col] = label_encoders[col].transform(input_df[col])
    if "Age" in input_df.columns:
        input_df["Age"] = scaler.transform(input_df[["Age"]])

    # Predict
    pred_prob = model.predict_proba(input_df)[0]
    pred_class = model.predict(input_df)[0]
    inv_label = label_encoders["Recurred"].inverse_transform([pred_class])[0]

    # Display results
    st.markdown("### 🎯 Prediction Result")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.success(f"🧬 Predicted Class: **{inv_label}**")
    with col2:
        st.info(f"📊 Probability: No Recurrence `{pred_prob[0]:.2f}` | Recurrence `{pred_prob[1]:.2f}`")

    # 📌 New Conclusion (bold, large, centered, no "Conclusion:" prefix)
    recurrence_prob = pred_prob[1]
    percentage = recurrence_prob * 100

    st.markdown(f"""
    <div style='padding-top:10px; font-size:34px; color:#2c3e50; text-align: center; font-weight: bold;'>
    <b>Based on feature values, predicted possibility of TC recurrence is {percentage:.2f}%</b>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------
    # SHAP Explanation (Compact stacked)
    # --------------------------
    with st.expander("🔍 Show SHAP Explanation (Class: Recurrence)"):
        explainer = shap.Explainer(model.predict_proba, X_train)
        shap_values = explainer(input_df)

        st.markdown("**📌 SHAP Force Plot**", unsafe_allow_html=True)
        st_shap(shap.plots.force(
            base_value=shap_values.base_values[0][1],
            shap_values=shap_values.values[0][:, 1],
            features=input_df.iloc[0],
            feature_names=input_df.columns
        ), height=80)

        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

        st.markdown("**📊 SHAP Waterfall Plot**", unsafe_allow_html=True)
        # 设置图形尺寸（宽度, 高度），单位为英寸
        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = 'DejaVu Serif'  # 或 "Microsoft YaHei" 等
        plt.rcParams['font.size'] = 4        # 设置字号
        plt.figure(figsize=(12, 10))
        st_shap(shap.plots.waterfall(shap_values[0, :, 1], max_display=len(features)), height=720)
        #plt.tight_layout()
        plt.tight_layout(pad=0.8)
        plt.show()

