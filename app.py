import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Apply custom CSS for enhanced styling
st.markdown("""
    <style>
    .stButton > button {
        color: white;
        background-color: #4CAF50;
        border: none;
        border-radius: 5px;
        font-size: 18px;
    }
    .st-expander {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e6e6e6;
    }
    .title {
        text-align: center;
        font-weight: bold;
        font-size: 32px;
        color: #2E8B57;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)
model=joblib.load('pipeline.pkl')
st.title("Parkinsons Disease Prediction")

with st.expander("Patient Information", expanded=True):

#Demographic details
    st.header("Demographic details")
    age=st.slider("Age",min_value=50,max_value=90,value=65)
    gender=st.selectbox("Gender",['Male','Female'])
    ethnicity=st.selectbox("Ethnicity",['Caucasian','African American','Asian','Other'])
    education=st.selectbox("Education Level",['None','High school','Bachelor\s','Higher'])
# Lifestyle Factors
with st.expander("Lifestyle Factors"):
    bmi = st.slider("BMI", min_value=15.0, max_value=40.0, value=25.0)
    smoking = st.selectbox("Smoking Status", ['No', 'Yes'])
    alcohol = st.slider("Alcohol Consumption (units per week)", min_value=0, max_value=20, value=5)
    physical_activity = st.slider("Physical Activity (hours per week)", min_value=0, max_value=10, value=2)
    diet_quality = st.slider("Diet Quality (0-10)", min_value=0, max_value=10, value=7)
    sleep_quality = st.slider("Sleep Quality (4-10)", min_value=4, max_value=10, value=6)

# Medical History
with st.expander("Medical History"):
    family_history = st.selectbox("Family History of Parkinson's", ['No', 'Yes'])
    tbi = st.selectbox("Traumatic Brain Injury", ['No', 'Yes'])
    hypertension = st.selectbox("Hypertension", ['No', 'Yes'])
    diabetes = st.selectbox("Diabetes", ['No', 'Yes'])
    depression = st.selectbox("Depression", ['No', 'Yes'])
    stroke = st.selectbox("History of Stroke", ['No', 'Yes'])

# Clinical Measurements
with st.expander("Clinical Measurements"):
    systolic_bp = st.slider("Systolic BP (mmHg)", min_value=90, max_value=180, value=120)
    diastolic_bp = st.slider("Diastolic BP (mmHg)", min_value=60, max_value=120, value=80)
    cholesterol_total = st.slider("Total Cholesterol (mg/dL)", min_value=150, max_value=300, value=200)
    cholesterol_ldl = st.slider("LDL Cholesterol (mg/dL)", min_value=50, max_value=200, value=100)
    cholesterol_hdl = st.slider("HDL Cholesterol (mg/dL)", min_value=20, max_value=100, value=50)
    cholesterol_triglycerides = st.slider("Triglycerides (mg/dL)", min_value=50, max_value=400, value=150)

# Cognitive and Functional Assessments
with st.expander("Cognitive and Functional Assessments"):
    updrs = st.slider("UPDRS Score", min_value=0, max_value=199, value=30)
    moca = st.slider("MoCA Score", min_value=0, max_value=30, value=25)
    functional_assessment = st.slider("Functional Assessment Score", min_value=0, max_value=10, value=7)

# Symptoms
with st.expander("Symptoms"):
    tremor = st.selectbox("Tremor", ['No', 'Yes'])
    rigidity = st.selectbox("Muscle Rigidity", ['No', 'Yes'])
    bradykinesia = st.selectbox("Bradykinesia", ['No', 'Yes'])
    postural_instability = st.selectbox("Postural Instability", ['No', 'Yes'])
    speech_problems = st.selectbox("Speech Problems", ['No', 'Yes'])
    sleep_disorders = st.selectbox("Sleep Disorders", ['No', 'Yes'])
    constipation = st.selectbox("Constipation", ['No', 'Yes'])

def convert_to_nan(value):
    return np.nan if value is None or value in ['Unknown', 'Prefer not to say'] else value
# Prepare input data for prediction
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [1 if gender == 'Female' else 0],
    'Ethnicity': [ethnicity],
    'EducationLevel': [education],
    'BMI': [bmi],
    'Smoking': [1 if smoking == 'Yes' else 0],
    'AlcoholConsumption': [alcohol],
    'PhysicalActivity': [physical_activity],
    'DietQuality': [diet_quality],
    'SleepQuality': [sleep_quality],
    'FamilyHistoryParkinsons': [1 if family_history == 'Yes' else 0],
    'TraumaticBrainInjury': [1 if tbi == 'Yes' else 0],
    'Hypertension': [1 if hypertension == 'Yes' else 0],
    'Diabetes': [1 if diabetes == 'Yes' else 0],
    'Depression': [1 if depression == 'Yes' else 0],
    'Stroke': [1 if stroke == 'Yes' else 0],
    'SystolicBP': [systolic_bp],
    'DiastolicBP': [diastolic_bp],
    'CholesterolTotal': [cholesterol_total],
    'CholesterolLDL': [cholesterol_ldl],
    'CholesterolHDL': [cholesterol_hdl],
    'CholesterolTriglycerides': [cholesterol_triglycerides],
    'UPDRS': [updrs],
    'MoCA': [moca],
    'FunctionalAssessment': [functional_assessment],
    'Tremor': [1 if tremor == 'Yes' else 0],
    'Rigidity': [1 if rigidity == 'Yes' else 0],
    'Bradykinesia': [1 if bradykinesia == 'Yes' else 0],
    'PosturalInstability': [1 if postural_instability == 'Yes' else 0],
    'SpeechProblems': [1 if speech_problems == 'Yes' else 0],
    'SleepDisorders': [1 if sleep_disorders == 'Yes' else 0],
    'Constipation': [1 if constipation == 'Yes' else 0]
})

# Button to make prediction
if st.button('Predict'):
    try:
        # Ensure input data is not empty
        if input_data.isnull().all(axis=1).any():
            st.error("Please fill in at least one feature to make a prediction.")
        else:
            # Get prediction and probability
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)

        
            proba_parkinsons = prediction_proba[0][1]  # Probability of class 1 (Parkinson's)
            
            # Display results
            if prediction[0] == 1:
                st.write("### The patient is predicted to have Parkinson's Disease.")
                st.write(f"#### Chance of having Parkinson's Disease: {proba_parkinsons * 100:.2f}%")
            else:
                st.write("### The patient is predicted NOT to have Parkinson's Disease.")
                st.write(f"#### Chance of having Parkinson's Disease: {proba_parkinsons * 100:.2f}%")

    except ValueError as e:
        st.error(f"Error in prediction: {e}. Please ensure valid data is entered.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
