import numpy as np
import pandas as pd
import joblib
import streamlit as st
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="TyG-related indices and SIRI–based stroke risk model",
    layout="wide",
)

FEATURES = ["TyG", "TyG_WC", "TyG_WHtR", "TyG_BMI", "SIRI"]

st.markdown(
    """
    <style>
    .big-risk-number {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .risk-level-text {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_explainer():
    model = joblib.load("RandomForest_final.pkl")

    preprocessor = None
    if isinstance(model, Pipeline):
        final_est = model.steps[-1][1]
        if len(model.steps) > 1:
            preprocessor = model[:-1]
    else:
        final_est = model

    explainer = shap.TreeExplainer(final_est)
    return model, explainer, preprocessor


model, explainer, preprocessor = load_model_and_explainer()

left_col, main_col = st.columns([1.1, 2.4])

with left_col:
    st.markdown("### Model information")
    st.markdown(
        """
- **Algorithm:** Random Forest classifier (within a pipeline)  
- **Outcome:** Stroke (yes / no)  
- **Inputs:** TyG-related indices and SIRI  

*This tool is for research use only and should not directly guide individual treatment decisions.*
        """
    )

    st.markdown("### Feature description")
    st.markdown(
        """
- **TyG** = ln[TG (mg/dL) × FBG (mg/dL) / 2]  
- **TyG_BMI** = TyG × BMI  
- **TyG_WC** = TyG × waist circumference  
- **TyG_WHtR** = TyG × (waist / height)  
- **SIRI** = (neutrophil × monocyte) / lymphocyte
        """
    )

with main_col:
    st.title("TyG-related indices and SIRI–based stroke risk prediction tool")

    st.markdown("Please input the patient's indicators:")

    c1, c2, c3 = st.columns(3)

    with c1:
        tyg = st.number_input("TyG", min_value=6.0, max_value=12.0, value=8.5, step=0.01)
        tyg_wc = st.number_input(
            "TyG_WC", min_value=600.0, max_value=1500.0, value=950.0, step=1.0
        )

    with c2:
        tyg_whtr = st.number_input(
            "TyG_WHtR", min_value=3.0, max_value=9.0, value=6.0, step=0.01
        )
        tyg_bmi = st.number_input(
            "TyG_BMI", min_value=150.0, max_value=450.0, value=280.0, step=0.5
        )

    with c3:
        siri = st.number_input(
            "SIRI", min_value=0.0, max_value=30.0, value=1.50, step=0.01
        )

    X_input = pd.DataFrame(
        [[tyg, tyg_wc, tyg_whtr, tyg_bmi, siri]],
        columns=FEATURES,
    )

    if st.button("Predict"):
        proba = float(model.predict_proba(X_input)[0, 1])

        st.markdown("### Prediction results")
        st.markdown(
            f"<p class='big-risk-number'>Estimated stroke risk: {proba * 100:.1f}%</p>",
            unsafe_allow_html=True,
        )

        if proba < 0.50:
            level = "Low risk (<0.50)"
            color_emoji = "🟦"
        elif proba < 0.75:
            level = "Intermediate risk (0.50–0.75)"
            color_emoji = "🟧"
        else:
            level = "High risk (≥0.75)"
            color_emoji = "🟥"

        st.markdown(
            f"<p class='risk-level-text'>Risk level: {color_emoji} <strong>{level}</strong></p>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### Model explanation for this patient (SHAP)")

        if preprocessor is not None:
            X_for_shap = preprocessor.transform(X_input)
        else:
            X_for_shap = X_input.values

        shap_values_all = explainer.shap_values(X_for_shap)

        if isinstance(shap_values_all, (list, tuple)):
            shap_sample = np.array(shap_values_all[1])[0]
        else:
            shap_arr = np.array(shap_values_all)
            if shap_arr.ndim == 2:
                shap_sample = shap_arr[0]
            elif shap_arr.ndim == 3 and shap_arr.shape[-1] == 2:
                shap_sample = shap_arr[0, :, 1]
            else:
                shap_sample = shap_arr[0]

        order = np.argsort(np.abs(shap_sample))[::-1]
        feat_sorted = np.array(FEATURES)[order]
        shap_sorted = shap_sample[order]

        top_text = ", ".join(
            [
                f"{name} ({val:+.3f})"
                for name, val in zip(feat_sorted[:3], shap_sorted[:3])
            ]
        )

        st.write(
            f"In this patient, the main drivers of predicted stroke risk are: **{top_text}** "
            "(SHAP value; positive values increase the risk, while negative values indicate a risk-lowering effect)."
        )

        colors = ["#d62728" if v > 0 else "#1f77b4" for v in shap_sorted]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(feat_sorted, shap_sorted, color=colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("SHAP value (impact on stroke risk for this patient)")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(
            "Red bars represent features pushing the model towards higher stroke risk (positive SHAP values), "
            "while blue bars indicate features that lower the predicted risk (negative SHAP values)."
        )

