import streamlit as st 
import numpy as np
import pandas as pd
import joblib

from feature_interpretation import FEATURE_INTERPRETATION

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Cardiac Risk Decision Support System",
    layout="wide"
)

# ===============================
# Global styling
# ===============================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    
    p {
    font-size: 1.3rem !important;
    }

    .risk-box {
        border: 2px solid #2ecc71;
        border-radius: 8px;
        padding: 2.5rem;
        height: 167px;
        text-align: center;
        font-size: 2.4rem;
        font-weight: bold;
        color: #2ecc71;
    }

    .risk-box-moderate {
        border: 2px solid #f39c12;
        border-radius: 8px;
        padding: 2.5rem;
        height: 167px;
        text-align: center;
        font-size: 2.4rem;
        font-weight: bold;
        color: #f39c12;
    }

    .risk-box-high {
        border: 2px solid #e74c3c;
        border-radius: 8px;
        padding: 2.5rem;
        height: 167px;
        text-align: center;
        font-size: 2.4rem;
        font-weight: bold;
        color: #e74c3c;
    }

    .prob-box {
        border: 2px solid #dddddd;
        border-radius: 8px;
        height: 167px;
        padding: 2rem 0rem 2rem 0rem;
        text-align: center;
    }

    html, body, [class*="css"] {
        font-size: 18px !important;
    }
    
    /* Headings */
    h1 {
        font-size: 2.6rem !important;
    }
    h2 {
        font-size: 2.1rem !important;
    }
    h3 {
        font-size: 1.7rem !important;
        padding: 1.70rem 0px 1rem !important;
    }
    
    /* Streamlit captions */
    .stCaption {
        font-size: 1.05rem !important;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        font-size: 2.4rem !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
    }
    
    /* Selectbox container */
    .stSelectbox > div > div {
        font-size: 1.3rem !important;
        height: 3.5rem !important;
        display: flex;
        align-items: center;
    }
    
    /* Dropdown options */
    div[role="listbox"] {
        font-size: 1.3rem !important;
    }
    
    .insight-box {
        border: 2px solid #dddddd;
        border-radius: 8px;
        padding: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# Load assets
# ===============================
@st.cache_resource
def load_assets():
    model = joblib.load("models/catboost_model.pkl")

    X_train = joblib.load("data/X_train.pkl")
    X_test = joblib.load("data/X_test.pkl")

    shap_train = joblib.load("explainability/shap_values_train.pkl")
    shap_test = joblib.load("explainability/shap_values_test.pkl")

    return model, X_train, X_test, shap_train, shap_test


cat_model, X_train, X_test, shap_train, shap_test = load_assets()

# Combine train + test for UI inference
X_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

shap_all = np.vstack([shap_train, shap_test])

# Remove last SHAP column (expected value)
# Remove last SHAP column (expected value)
shap_all = shap_all[:, :-1]
feature_names = X_all.columns.tolist()


# ===============================
# Title & description
# ===============================
st.title("🫀 Cardiac Risk Decision Support System")

st.caption(
    "Explainable AI–assisted cardiac rehabilitation risk stratification "
    "(for clinical decision support only)"
)

st.divider()

# ===============================
# Patient selection
# ===============================
st.markdown("## 🔍 Patient Selection")
patient_idx = st.selectbox(
    "List of patients from dataset",
    options=list(range(len(X_all))),
    format_func=lambda x: f"Patient {x}"
)

st.divider()

# ===============================
# MOCK prediction logic (Phase 1)
# This will later be replaced by CatBoost
# ===============================
patient_X = X_all.iloc[[patient_idx]]
risk_prob = cat_model.predict_proba(patient_X)[0, 1]

if risk_prob < 0.4:
    risk_label = "LOW CARDIAC RISK"
    risk_class = "risk-box"
else:
    risk_label = "MODERATE-HIGH CARDIAC RISK"
    risk_class = "risk-box-high"

# ===============================
# Risk display row
# ===============================
left_col, right_col = st.columns([3, 1])

with left_col:
    st.markdown(
        f"""
        <div class="{risk_class}">
            {risk_label}
        </div>
        """,
        unsafe_allow_html=True
    )

with right_col:
    st.markdown(
        f"""
        <div class="prob-box">
            <div style="font-size: 1.5rem; color: #777;">
                Predicted Risk Probability
            </div>
            <div style="font-size: 2rem; font-weight: bold;">
                {risk_prob:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===============================
# Insights + Key Factors (Split Layout)
# ===============================
st.divider()

left_col, right_col = st.columns([2, 1])

# ===============================
# Feature display configuration
# ===============================

FEATURE_DISPLAY_NAME = {
    "Heart Rate (%)": "Heart Rate",
    "Test Today - METS": "Exercise Capacity",
    "Exercise Habit - Duration": "Exercise Duration",
    "Exercise Habit - Frequency": "Exercise Frequency",
    "exercise_any": "Regular Physical Activity",
    "Age": "Age",
    "ecg_q_wave": "ECG Q Wave",
    "dx_pci": "History of PCI",
    "dx_cabg": "History of CABG",
    "muscle_power_left": "Muscle Strength",
    "ecg_sinus_rhythm": "ECG Sinus Rhythm",
    "termination_adverse": "Adverse Test Termination",
    "Test Today - peak HR": "Peak Heart Rate During Test"
}

ICON_MAP = {
    "Heart Rate (%)": "❤️",
    "Test Today - METS": "🏃",
    "Exercise Habit - Duration": "⏱️",
    "Exercise Habit - Frequency": "📅",
    "exercise_any": "🧘",
    "Age": "🎂",
    "ecg_q_wave": "📈",
    "ecg_sinus_rhythm":"📈",
    "dx_pci": "🏥",
    "dx_cabg": "🏥",
    "muscle_power_left": "💪",
    "termination_adverse": "⚠️",
    "Test Today - peak HR": "❤️‍🔥"
}

# ======================================================
# LEFT COLUMN — MODEL INSIGHTS
# ======================================================
with left_col:


    st.markdown("## 🔍 Model Insights")
    st.caption("Key factors influencing the model’s risk assessment for this patient")

    patient_shap = shap_all[patient_idx]
    # Sort features by absolute SHAP impact
    sorted_idx = np.argsort(np.abs(patient_shap))[::-1]

    # Split into protective vs residual risk factors
    protective_idx = [i for i in sorted_idx if patient_shap[i] < 0][:3]
    risk_idx = [i for i in sorted_idx if patient_shap[i] > 0][:3]

    # ===============================
    # Summary
    # ===============================
    if risk_prob < 0.4:
        summary_text = (
            "Overall cardiac risk is low because strong protective factors "
            "outweigh existing risk contributors."
        )
        summary_border = "#2ecc71"
    elif risk_prob < 0.6:
        summary_text = (
            "Cardiac risk is moderate due to a balance between protective "
            "and risk-increasing factors."
        )
        summary_border = "#f39c12"
    else:
        summary_text = (
            "Cardiac risk is high due to multiple dominant risk-increasing factors."
        )
        summary_border = "#e74c3c"

    st.markdown(
        f"""
        <div style="
            background-color: #f5f7fa;
            border-left: 6px solid {summary_border};
            padding: 0.8rem;
            min-height: 80px;
            border-radius: 6px;
            margin-bottom: 0.2rem;
            color: #111;
            font-size: 1.5rem;
        ">
            <strong>Summary:</strong> {summary_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ===============================
    # Protective Factors
    # ===============================
    if protective_idx:
        st.markdown("### 🟢 Protective Factors")

        for i in protective_idx:
            feature = feature_names[i]
            explanation = FEATURE_INTERPRETATION.get(
                feature, {}
            ).get(
                "negative",
                f"{FEATURE_DISPLAY_NAME.get(feature, feature)} reduced predicted cardiac risk"
            )

            st.markdown(
                f"""
                <div style="
                    background-color: #f3fff7;
                    border-left: 5px solid #2ecc71;
                    padding: 0.8rem;
                    min-height: 80px;
                    border-radius: 6px;
                    margin-bottom: 0.6rem;
                    color: #111;
                    font-size: 1.5rem;
                    line-height: 1.5;
                ">
                    {explanation}
                </div>
                """,
                unsafe_allow_html=True
            )

    # ===============================
    # Residual Risk Factors
    # ===============================
    if risk_idx:
        st.markdown("### 🔴 Residual Risk Factors")

        for i in risk_idx:
            feature = feature_names[i]
            explanation = FEATURE_INTERPRETATION.get(
                feature, {}
            ).get(
                "positive",
                f"{FEATURE_DISPLAY_NAME.get(feature, feature)} increased predicted cardiac risk"
            )

            st.markdown(
                f"""
                <div style="
                    background-color: #fff5f5;
                    border-left: 5px solid #e74c3c;
                    padding: 0.9rem;
                    min-height: 80px;
                    border-radius: 6px;
                    margin-bottom: 0.6rem;
                    color: #111;
                    font-size: 1.5rem;
                    line-height: 1.5;
                ">
                    {explanation}
                </div>
                """,
                unsafe_allow_html=True
            )

    # Close outer container
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# RIGHT COLUMN — KEY FACTORS TILES
# ======================================================
with right_col:

    # Top 4 features
    top_tiles_idx = sorted_idx[:4]

    for i in top_tiles_idx:
        raw_feature = feature_names[i]
        shap_val = patient_shap[i]

        display_name = FEATURE_DISPLAY_NAME.get(raw_feature, raw_feature)
        icon = ICON_MAP.get(raw_feature, "🔎")

        if shap_val < 0:
            status = "Protective"
            bg = "#f3fff7"
            border = "#2ecc71"
            text_color = "#1b5e20"
        else:
            status = "Needs Attention"
            bg = "#fff5f5"
            border = "#e74c3c"
            text_color = "#7f0000"

        st.markdown(
            f"""
            <div style="
                background-color: {bg};
                border-top: 6px solid {border};
                border-radius: 8px;
                padding: 1.1rem;
                min-height: 150px;
                text-align: center;
                color: {text_color};
                margin-bottom: 0.8rem;
            ">
                <div style="font-size: 2.1rem;">{icon}</div>
                <div style="font-weight: 600; margin-top: 0.2rem;font-size: 2.0rem;">
                    {display_name}
                </div>
                <div style="margin-top: 0.5rem; font-size: 1.5rem;">
                    {status}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ===============================
# Footer
# ===============================
st.divider()
st.caption(
    "⚠️ This system is a clinical decision-support tool only. "
    "Predictions are not medical advice and must be interpreted by qualified professionals."
)
