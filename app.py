import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap

# 1. Load the trained model
@st.cache_resource
def load_model():
    with open('student_placement_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# 2. Your Exact Feature Engineering Function
def build_student_features(student_df):
    student_df = student_df.sort_values("Semester").copy()
    sgpa = student_df["SGPA"].values
    sems = student_df["Semester"].values
    
    expanding_mean = student_df['SGPA'].expanding().mean()
    expanding_std = student_df['SGPA'].expanding().std().fillna(0)
    is_dip = student_df['SGPA'] < (expanding_mean - expanding_std)
    
    # Slop for Trend
    if len(sgpa)>1:
        polyfit = np.polyfit(sems, sgpa, 1)
        slope = polyfit[0]
    else:
        slope = 0.0

    return pd.Series({
        'sgpa_mean' : sgpa.mean(),
        "sgpa_trend" : round(slope, 4),
        'consistency_score' : sgpa.std(),
        'attendance_mean': student_df['Attendance'].mean(),
        'internals_avg':     student_df['Internals'].mean()
    })

# 3. Your Exact SHAP Explanation Function
def explain_student(student_df, feature_list, model):
    explainer = shap.TreeExplainer(model)
    engineered = build_student_features(student_df)
    input_df = pd.DataFrame([engineered])[feature_list]

    # New SHAP API
    explanation = explainer(input_df)
    values = explanation.values

    # Handle binary classifier output
    if values.ndim == 3:
        # (samples, features, classes)
        contribs = values[0, :, 1]
    else:
        # (samples, features)
        contribs = values[0]

    return pd.Series(contribs, index=feature_list).sort_values()

PRESCRIPTIONS = {
    "internals_avg": "Internal assessment scores are dragging your placement chances down. Focus on quizzes/assignments — internals are the single strongest predictor in the model.",
    "sgpa_trend": "Your SGPA trend is negative — grades are declining semester over semester. Stabilizing (even without a big jump) matters more than one great semester.",
    "current_sgpa": "Your most recent semester's SGPA is a weak point relative to your history.",
    "dip_count": "You've had several dip semesters (dropping below your rolling average). Consistency matters — avoid volatility even if the average looks fine.",
    "consistency_score": "Your SGPA is volatile across semesters. Predictability helps more than occasional high peaks.",
    "attendance_mean": "Attendance is below where it needs to be — this correlates with the other risk factors even if it's not the top driver.",
    "last_sem_dip": "Your most recent semester was itself a dip — this is a recency-weighted red flag."
}

# NEW: Add the advice generator
def generate_advice(contribs, top_n=2):
    negative_contribs = contribs[contribs < 0].sort_values()
    top_issues = negative_contribs.head(top_n).index.tolist()
    return [PRESCRIPTIONS[f] for f in top_issues if f in PRESCRIPTIONS]

# 4. Streamlit UI
st.set_page_config(page_title="Placement Predictor", layout="wide")
st.title("🎓 Student Placement Readiness Predictor")
st.write("Enter the student's semester-by-semester records below.")

# Interactive table for raw data input
n_semesters = st.number_input("Number of completed semesters:", min_value=1, max_value=8, value=1)

default_data = pd.DataFrame({
    "Semester": list(range(1, int(n_semesters) + 1)),
    "SGPA": [5.0] * int(n_semesters),
    "Attendance": [0.0] * int(n_semesters),
    "Internals": [0.0] * int(n_semesters)
})

st.write("### Edit Student Record")
edited_df = st.data_editor(default_data, hide_index=True, use_container_width=True)

# 5. Prediction Execution
FEATURES_TRIMMED = [
    "internals_avg", "sgpa_mean",
    "sgpa_trend", "consistency_score", "attendance_mean"
]

if st.button("Predict Readiness", type="primary"):
    # Convert raw table to the 5 features
    engineered_features = build_student_features(edited_df)
    input_df = pd.DataFrame([engineered_features])[FEATURES_TRIMMED]
    
    # Predict
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    
    st.markdown("---")
    
    # Display Layout
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Prediction Result")
        if prediction == "Placed" or prediction == "Placement Ready":
            st.success(f"🎉 **Placement Ready**")
        else:
            st.error(f"⚠️ **{prediction}**")
            
        st.metric(label="Probability of Placement", value=f"{max(probabilities) * 100:.2f}%")
        
        with st.expander("View Calculated Background Features"):
            st.dataframe(input_df.T, use_container_width=True)
            
    with col2:
            st.subheader("What drove this prediction?")
            st.caption("Positive values push the model toward 'Ready', negative push toward 'Not Ready'.")
            
            # Get SHAP values using your function
            contribs = explain_student(edited_df, FEATURES_TRIMMED, model)
            
            # Display as a bar chart natively in Streamlit
            st.bar_chart(contribs, horizontal=True)
            
            # NEW: Generate and display the advice
            advice_list = generate_advice(contribs)
            
            if advice_list:
                st.subheader("💡 Actionable Feedback")
                for advice in advice_list:
                    # Using st.warning makes it stand out visually as an area for improvement
                    st.warning(advice)