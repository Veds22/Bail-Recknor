import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from utils import preprocessing_pipeline, predict_bail_score
import json
import joblib


# -------------------------------
# Load Model & Artifacts
# -------------------------------

@st.cache_resource
def load_artifacts():
    """Load model and required artifacts with caching"""
    try:
        mlp = load_model('./artefacts/mlp_model.h5')
        
        with open("artefacts/ipc_sections.json", "r") as f:
            ipc_data = json.load(f)
        
        # Load feature columns to ensure proper alignment
        try:
            feature_columns = joblib.load('./artefacts/feature_columns.pkl')
        except FileNotFoundError:
            st.warning("⚠️ feature_columns.pkl not found. Feature alignment may fail.")
            feature_columns = None
        
        return mlp, ipc_data["ipc_sections_list"], feature_columns
    
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {e}")
        st.stop()


mlp, ipc_section_options, feature_columns = load_artifacts()


# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(page_title="Bail Reckoner", layout="centered", page_icon="⚖️")

st.title("⚖️ Bail Reckoner – Case Input Form")
st.markdown("Fill in the case details below to predict bail likelihood:")

# -------------------------------
# Dropdown values
# -------------------------------

bail_type_options = ['Regular', 'Anticipatory', 'Interim', 'Unknown', 'Others', 'Not applicable']
gender_options = ['Male', 'Female', 'Unknown', 'Multiple']
prior_cases_options = ['No', 'Yes', 'Unknown']
crime_type_options = ['Murder', 'Sexual Offense', 'Fraud or Cheating', 'Narcotics', 'Others']
region_options = ['Assam', 'Tamil Nadu', 'Kerala', 'West Bengal', 'Jammu & Kashmir', 'Punjab',
 'Karnataka', 'Maharashtra', 'Himachal Pradesh', 'Uttar Pradesh', 'Gujarat',
 'Madhya Pradesh', 'Haryana', 'Delhi', 'Uttarakhand', 'Bihar', 'Manipur',
 'Chhattisgarh', 'Odisha', 'Tripura', 'Rajasthan', 'Telangana', 'Jharkhand',
 'Andhra Pradesh', 'Ladakh', 'Chandigarh', 'Puducherry', 'Punjab and Haryana']
court_level_map = {
    "Supreme Court": 1,
    "High Court": 2,
    "District Court": 3
}

# -------------------------------
# Input Form
# -------------------------------

with st.form("bail_form"):
    
    col1, col2 = st.columns(2)
    
    with col1:
        bail_type = st.selectbox("Bail Type", bail_type_options)
        accused_gender = st.selectbox("Accused Gender", gender_options)
        prior_cases = st.selectbox("Prior Criminal Cases", prior_cases_options)
        crime_type = st.selectbox("Crime Type", crime_type_options)
    
    with col2:
        region = st.selectbox("Region", region_options)
        court_level_label = st.selectbox("Court Level", list(court_level_map.keys()))
        court_level = court_level_map[court_level_label]
    
    st.markdown("---")
    
    ipc_sections = st.multiselect(
        label="IPC Sections Involved",
        options=ipc_section_options,
        help="Select all applicable IPC sections"
    )
    
    st.markdown("**Case Attributes:**")
    col3, col4 = st.columns(2)
    
    with col3:
        bail_cancellation_case = st.checkbox("Bail Cancellation Case")
        landmark_case = st.checkbox("Landmark Case Cited")
    
    with col4:
        parity_argument_used = st.checkbox("Parity Argument Used")
        bias_flag = st.checkbox("Potential Bias Flag")
    
    facts = st.text_area(
        "Case Facts",
        placeholder="Enter the factual background of the case...",
        height=150
    )
    
    submitted = st.form_submit_button("🔍 Predict Bail Outcome", use_container_width=True)

# -------------------------------
# Prediction Logic
# -------------------------------

if submitted:
    
    # Validate inputs
    if not facts.strip():
        st.error("⚠️ Please provide case facts before submitting.")
        st.stop()
    
    if not ipc_sections:
        st.warning("⚠️ No IPC sections selected. Proceeding with empty IPC list.")
    
    # Prepare input data
    input_data = {
        "ipc_sections": ipc_sections,
        "bail_type": bail_type,
        "bail_cancellation_case": bail_cancellation_case,
        "landmark_case": landmark_case,
        "accused_gender": accused_gender,
        "prior_cases": prior_cases,
        "crime_type": crime_type,
        "facts": facts,
        "bias_flag": bias_flag,
        "parity_argument_used": parity_argument_used,
        "region": region,
        "court_level": court_level,
    }
    
    with st.spinner("Processing case data..."):
        try:
            input_df = pd.DataFrame([input_data])

            # Preprocess
            input_df_processed = preprocessing_pipeline(input_df)

            # Align to training schema
            if feature_columns is not None:
                for col in feature_columns:
                    if col not in input_df_processed.columns:
                        input_df_processed[col] = 0

                input_df_processed = input_df_processed[feature_columns]

            # 🔒 Feature-size validation
            model_expected_features = mlp.input_shape[1]
            processed_features = input_df_processed.shape[1]

            if processed_features != model_expected_features:
                st.error("❌ Feature mismatch detected")
                st.stop()

            # Predict
            prediction = predict_bail_score(input_df_processed, mlp)

            bail_score = float(prediction[0][0]) if prediction.ndim > 1 else float(prediction[0])

            st.success("✅ Case processed successfully!")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.exception(e)
            st.stop()

    # -------------------------------
    # Display Results
    # -------------------------------
    
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Score visualization
    col_score1, col_score2, col_score3 = st.columns([1, 2, 1])
    
    with col_score2:
        st.metric(
            label="Bail Likelihood Score",
            value=f"{bail_score:.2%}",
            delta="High Confidence" if abs(bail_score - 0.5) > 0.3 else "Moderate Confidence"
        )
    
    # Verdict
    st.markdown("### Verdict:")
    
    if bail_score > 0.5:
        st.success(f"### ✅ Bail Likely to be **GRANTED**")
        st.progress(bail_score)
        st.info(f"The model predicts a **{bail_score:.1%}** probability of bail being granted based on the provided case details.")
    else:
        st.error(f"### ❌ Bail Likely to be **REJECTED**")
        st.progress(bail_score)
        st.info(f"The model predicts a **{(1-bail_score):.1%}** probability of bail being rejected based on the provided case details.")
    
    # -------------------------------
    # Optional: Show processed features
    # -------------------------------
    
    with st.expander("🔍 View Processed Features (Debug)"):
        st.write(f"**Total Features:** {input_df_processed.shape[1]}")
        st.dataframe(input_df_processed.head())
    
    # -------------------------------
    # Disclaimer
    # -------------------------------
    
    st.markdown("---")
    st.caption("""
    ⚠️ **Disclaimer:** This prediction is generated by a machine learning model and should not be considered 
    legal advice. Always consult with a qualified legal professional for case-specific guidance.
    """)