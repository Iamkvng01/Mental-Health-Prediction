import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessing info
rf_model = joblib.load("mental_health_rf_model.pkl")
feature_cols = joblib.load("feature_columns.pkl")
feature_dtypes = joblib.load("feature_dtypes.pkl")

# --- Helper function ---
def predict_student(model, feature_cols, dtypes, student_info, course):
    student = pd.DataFrame([0]*len(feature_cols), index=feature_cols).T
    student = student.astype(dtypes)
    
    for key, value in student_info.items():
        if key in student.columns:
            student.loc[0, key] = value

    for col in feature_cols:
        if col.startswith("Course_"):
            student.loc[0, col] = False
    
    student.loc[0, f"Course_{course}"] = True

    prediction = model.predict(student)[0]
    proba = model.predict_proba(student)[0]
    
    return prediction, proba

# --- Feature descriptions ---
feature_help = {
    "Gender": "Select the student's gender. This may influence emotional coping styles and social support.",
    "Age": "The student's current age in years.",
    "Year of Study": "The student's current academic level.",
    "CGPA": "Cumulative Grade Point Average. Reflects overall academic performance.",
    "Academic Workload": "How demanding or heavy the student's academic tasks feel (1 = light, 10 = heavy).",
    "Academic Pressure": "The level of pressure the student feels from academic expectations.",
    "Financial Concerns": "How much financial stress the student is currently facing.",
    "Social Relationships": "How connected the student feels with peers, friends, and family.",
    "Average Sleep": "Average hours of sleep per night.",
    "Study Satisfaction": "How satisfied the student feels about their study experience.",
    "Anxiety": "General level of anxiety or nervousness.",
    "Isolation": "How socially isolated or disconnected the student feels.",
    "Future Insecurity": "How worried the student feels about their future (career, goals, etc.).",
    "Panic Attack": "Whether the student has experienced panic attacks.",
    "Risk Level": "Self-assessed emotional risk level.",
    "Course": "The student's course of study."
}

# --- Page setup ---
st.set_page_config(page_title="Student Mental Health Predictor", layout="wide")

# --- Custom CSS for Calm Health Theme ---
st.markdown("""
    <style>
        body {
            background-color: #F2FAFB;
        }
        .main {
            background-color: #FFFFFF;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0px 2px 12px rgba(0,0,0,0.08);
        }
        h1, h2, h3 {
            color: #1B4965;
        }
        .stProgress > div > div > div > div {
            background-color: #6BCB77;
        }
        .result-card {
            background-color: #E3F2FD;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            color: #1B4965;
            font-weight: 600;
        }
        footer {
            text-align: center;
            font-size: 0.9rem;
            color: gray;
            margin-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("Student Mental Health Prediction")
st.markdown(
    "An AI-powered tool designed to estimate **student depression risk levels** "
    "based on academic, social, and personal indicators."
)
st.info("This is a research and educational tool, not a medical diagnosis.")

# --- Sidebar Inputs ---
st.sidebar.header("Student Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"], help=feature_help["Gender"])
age = st.sidebar.number_input("Age", 16, 40, 21, help=feature_help["Age"])
year_of_study = st.sidebar.slider("Year of Study", 1, 7, 3, help=feature_help["Year of Study"])
cgpa = st.sidebar.number_input("CGPA", 0.0, 5.0, 3.0, step=0.1, help=feature_help["CGPA"])

academic_workload = st.sidebar.slider("Academic Workload", 1, 10, 5, help=feature_help["Academic Workload"])
academic_pressure = st.sidebar.slider("Academic Pressure", 1, 10, 5, help=feature_help["Academic Pressure"])
financial_concerns = st.sidebar.slider("Financial Concerns", 1, 10, 5, help=feature_help["Financial Concerns"])
social_relationships = st.sidebar.slider("Social Relationships", 1, 10, 5, help=feature_help["Social Relationships"])
average_sleep = st.sidebar.slider("Average Sleep (hours)", 1, 12, 6, help=feature_help["Average Sleep"])
study_satisfaction = st.sidebar.slider("Study Satisfaction", 1, 10, 5, help=feature_help["Study Satisfaction"])

anxiety = st.sidebar.slider("Anxiety Level", 0, 10, 2, help=feature_help["Anxiety"])
isolation = st.sidebar.slider("Isolation Level", 0, 10, 2, help=feature_help["Isolation"])
future_insecurity = st.sidebar.slider("Future Insecurity", 0, 10, 2, help=feature_help["Future Insecurity"])

panic_attack = st.sidebar.radio("Do You Experience Panic Attacks?", ["No", "Yes"], help=feature_help["Panic Attack"])
panic_attack = 1 if panic_attack == "Yes" else 0

risk_level = st.sidebar.radio("Risk Level (Self-assessment)", ["Low", "High"], help=feature_help["Risk Level"])
risk_level = 1 if risk_level == "High" else 0

course_options = [c.replace("Course_", "") for c in feature_cols if c.startswith("Course_")]
course = st.sidebar.selectbox("Course of Study", course_options, help=feature_help["Course"])

# --- Main section: Prediction ---
if st.sidebar.button(" Predict Depression Level"):
    student_info = {
        "Gender": 1 if gender == "Male" else 0,
        "Age": age,
        "Year of Study": year_of_study,
        "CGPA": cgpa,
        "Academic Workload": academic_workload,
        "Academic Pressure": academic_pressure,
        "Financial Concerns": financial_concerns,
        "Social Relationships": social_relationships,
        "Average Sleep": average_sleep,
        "Study Satisfaction": study_satisfaction,
        "Anxiety": anxiety,
        "Isolation": isolation,
        "Future Insecurity": future_insecurity,
        "Panic Attack": panic_attack,
        "Risk Level": risk_level
    }
    
    prediction, proba = predict_student(rf_model, feature_cols, feature_dtypes, student_info, course)

    # --- Robust prediction handling ---
    try:
        pred_scalar = prediction.item()
    except Exception:
        pred_scalar = prediction

    if isinstance(pred_scalar, (int, np.integer)):
        int_map = {0: "Low", 1: "Moderate", 2: "High"}
        level = int_map.get(int(pred_scalar), str(pred_scalar))
    else:
        level = str(pred_scalar)

    color_map = {"Low": "#6BCB77", "Moderate": "#FFD93D", "High": "#FF6B6B"}
    card_color = color_map.get(level, "#E3F2FD")

    st.markdown("---")
    st.subheader(" Prediction Result")
    st.markdown(
        f"<div class='result-card' style='background-color:{card_color};'>"
        f"<h3>Predicted Depression Level: {level}</h3>"
        "</div>", unsafe_allow_html=True
    )

    # --- Personalized Explanation ---
    explanation = []

    if average_sleep <= 5:
        explanation.append("limited sleep")
    if academic_pressure >= 7 or academic_workload >= 7:
        explanation.append("high academic workload or pressure")
    if isolation >= 6:
        explanation.append("increased isolation")
    if financial_concerns >= 7:
        explanation.append("financial stress")
    if social_relationships <= 4:
        explanation.append("low social connection")
    if anxiety >= 7:
        explanation.append("high anxiety levels")

    if explanation:
        factors = ", ".join(explanation[:-1]) + (" and " + explanation[-1] if len(explanation) > 1 else "")
        st.markdown(
            f"**Summary:** Based on your {factors}, your predicted depression level is **{level.lower()}**."
        )
    else:
        st.markdown(
            f"**Summary:** Your responses indicate a balanced state. The predicted depression level is **{level.lower()}**."
        )

    # --- Handle probabilities safely ---
    model_classes = list(rf_model.classes_)
    prob_dict = {cls: float(p) for cls, p in zip(model_classes, proba)}
    ordered_levels = ["Low", "Moderate", "High"]
    probs_ordered = [prob_dict.get(l, 0.0) for l in ordered_levels]

    st.markdown("### Probability Distribution")
    
    st.bar_chart(pd.DataFrame({
        "Depression Level": ordered_levels,
        "Probability": probs_ordered
    }).set_index("Depression Level"))

    # st.write("### Probabilities (detailed)")
    # for lvl, p in zip(ordered_levels, probs_ordered):
    #     st.write(f"- **{lvl}:** {p:.2f}")

# --- Footer ---
st.markdown("<footer>© 2025 Calm Health | Developed for Research Purposes Only</footer>", unsafe_allow_html=True)
