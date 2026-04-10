import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL + ENCODERS
# =========================
model = joblib.load("resource_model.pkl")
label_encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="AI Resource Allocator")

st.title("🚀 Smart Resource Allocation System")

# =========================
# USER INPUT
# =========================
skill_level = st.selectbox("Skill Level", ["Beginner", "Intermediate", "Expert"])
experience_years = st.slider("Experience", 0, 10, 3)

availability = st.selectbox("Availability", ["Available", "Busy"])
availability = 1 if availability == "Available" else 0

location = st.selectbox("Location", ["A", "B", "C"])
task_type = st.selectbox("Task Type", ["Delivery", "Maintenance", "Support"])
task_complexity = st.selectbox("Task Complexity", ["Low", "Medium", "High"])
required_skill = st.selectbox("Required Skill", ["Beginner", "Intermediate", "Expert"])

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict"):

    input_data = pd.DataFrame([{
        "skill_level": skill_level,
        "experience_years": experience_years,
        "availability": availability,
        "location": location,
        "task_type": task_type,
        "task_complexity": task_complexity,
        "required_skill": required_skill
    }])

    # Encode using saved encoders
    for col in ["skill_level", "location", "task_type", "task_complexity", "required_skill"]:
        input_data[col] = label_encoders[col].transform(input_data[col])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success("✅ Resource SHOULD be Allocated")
    else:
        st.error("❌ Resource should NOT be Allocated")

    st.write(f"Confidence Score: {probability:.2f}")