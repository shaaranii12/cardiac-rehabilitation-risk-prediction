import os
import numpy as np
import google.generativeai as genai
from feature_interpretation import FEATURE_INTERPRETATION

# ===============================
# Gemini setup
# ===============================
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. "
        "Set it as an environment variable."
    )

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")

# ===============================
# Helper: extract SHAP drivers
# ===============================
def extract_key_risk_drivers(
    shap_values,
    X,
    patient_idx,
    top_k=5
):
    shap_patient = shap_values[patient_idx]
    feature_names = X.columns

    top_idx = np.argsort(np.abs(shap_patient))[-top_k:][::-1]

    drivers = []
    for i in top_idx:
        feature = feature_names[i]
        shap_val = shap_patient[i]

        if feature in FEATURE_INTERPRETATION:
            direction = "positive" if shap_val > 0 else "negative"
            drivers.append(FEATURE_INTERPRETATION[feature][direction])
        else:
            direction = "increased" if shap_val > 0 else "reduced"
            drivers.append(f"{feature} {direction} predicted cardiac risk")

    return drivers

# ===============================
# Main report generator
# ===============================
def generate_full_clinical_model_report(
    patient_idx,
    X_train_cat,
    y_train_prob_cat,
    shap_values_A_only
):
    risk_prob = y_train_prob_cat[patient_idx]

    key_risk_drivers = extract_key_risk_drivers(
        shap_values=shap_values_A_only,
        X=X_train_cat,
        patient_idx=patient_idx
    )

    drivers_text = "\n".join([f"- {d}" for d in key_risk_drivers])

    prompt = f"""
You are a senior clinical machine learning analyst.

You are explaining the output of a cardiac rehabilitation
risk prediction model for transparency and auditability.

You do NOT diagnose, prescribe, or provide medical advice.

PATIENT CONTEXT:
- Patient ID: {patient_idx}
- Predicted probability of moderate-to-high cardiac risk: {risk_prob:.2f}

KEY MODEL-IDENTIFIED RISK DRIVERS:
{drivers_text}

TASK:
Write a clear, structured explainability report using
neutral clinical language and short paragraphs.

Use the EXACT structure below.

===========================
1. Executive Summary

2. Risk Prediction Outcome

3. Key Contributing Factors

4. Model Reliability and Generalisation

5. Limitations and Clinical Considerations

6. Conclusion
===========================

RULES:
- Do not introduce new clinical facts
- Do not make treatment recommendations
- Do not speculate beyond the provided drivers
- Keep explanations factual and cautious
"""

    response = model.generate_content(prompt)

    return response.text