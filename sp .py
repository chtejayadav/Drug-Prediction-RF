import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ✅ Ensure set_page_config() is the first Streamlit command
st.set_page_config(page_title="Patient Drug Prediction", layout="wide")

# Function to set background image and change text color
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    
    background_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    h1, h2, h3, h4, h5, h6, label, span {{
        color: white !important; /* Force white text */
    }}
    .stSelectbox div[data-testid="stMarkdownContainer"] * {{
        color: white !important; /* Keep dropdown values black !important*/
    }}
    /* Remove sidebar background */
    section[data-testid="stSidebar"] {{
        background-color: transparent !important;
        color: white !important;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Set background image
image_path = "bgg.gif"  # Ensure the image exists in the same directory
set_background(image_path)

# Load the trained model (for demonstration, we train again here)
def train_model():
    # Load dataset
    file_path = "sp.csv"
    df = pd.read_csv(file_path)
    df = df.drop(columns=["Patient_ID"])  # Drop non-essential column
    
    # Encode categorical variables
    label_encoders = {}
    for col in ["Gender", "Prescribed_Drug"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Define features and target
    X = df.drop(columns=["Prescribed_Drug"])
    y = df["Prescribed_Drug"]
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model and encoders
    with open("rf_model.pkl", "wb") as f:
        pickle.dump((model, label_encoders), f)

# Train and save the model initially
train_model()

# Load the trained model
with open("rf_model.pkl", "rb") as f:
    model, label_encoders = pickle.load(f)

# Load dataset again to get feature names
file_path = "sp.csv"
df = pd.read_csv(file_path)
df = df.drop(columns=["Patient_ID"])
feature_names = df.drop(columns=["Prescribed_Drug"]).columns

# Streamlit UI
st.title("💊 Patient Drug Prescription Prediction")
st.markdown("### Enter patient details to predict the prescribed drug.")

# Sidebar with About Section
with st.sidebar:
    st.header("🔍 About the App")
    st.markdown("""
    - **Purpose**: Predicts the prescribed drug based on patient details.  
    - **Technology Used**: Streamlit, Machine Learning (Random Forest), Pandas, Matplotlib, Seaborn.  
    - **Input Features**:  
      - Age, Gender  
      - Blood Sugar (Fasting & Post-Meal)  
      - HbA1c, BMI  
      - Blood Pressure (Systolic & Diastolic)  
      - Cholesterol Level  
    - **Model Training**: Uses a trained Random Forest model on patient data.  
    - **Prediction Output**: Displays the most likely prescribed drug.  
    - **Additional Feature**: Shows feature importance in drug prescription.  
    """)

# Create a form
with st.form("patient_form"):
    st.subheader("📝 Patient Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.selectbox("Age", options=list(range(0, 121)), index=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        fasting_blood_sugar = st.selectbox("Fasting Blood Sugar", options=list(range(50, 301)), index=50)
    
    with col2:
        post_meal_blood_sugar = st.selectbox("Post-Meal Blood Sugar", options=list(range(50, 501)), index=140)
        hba1c = st.selectbox("HbA1c", options=[round(x * 0.1, 1) for x in range(30, 151)], index=25)
        bmi = st.selectbox("BMI", options=[round(x * 0.1, 1) for x in range(100, 501)], index=120)
    
    with col3:
        blood_pressure_sys = st.selectbox("Systolic Blood Pressure", options=list(range(80, 201)), index=40)
        blood_pressure_dia = st.selectbox("Diastolic Blood Pressure", options=list(range(50, 131)), index=30)
        cholesterol_level = st.selectbox("Cholesterol Level", options=list(range(100, 301)), index=50)
    
    submit_button = st.form_submit_button("💊 Predict Prescription")

# Convert input into DataFrame
if submit_button:
    input_data = pd.DataFrame([[age, gender, fasting_blood_sugar, post_meal_blood_sugar, hba1c, bmi,
                                blood_pressure_sys, blood_pressure_dia, cholesterol_level]],
                              columns=["Age", "Gender", "Fasting_Blood_Sugar", "Post_Meal_Blood_Sugar", "HbA1c",
                                       "BMI", "Blood_Pressure_Sys", "Blood_Pressure_Dia", "Cholesterol_Level"])
    
    # Encode categorical feature
    input_data["Gender"] = label_encoders["Gender"].transform([gender])
    
    # Predict
    prediction = model.predict(input_data)[0]
    predicted_drug = label_encoders["Prescribed_Drug"].inverse_transform([prediction])[0]
    
    # Display prediction result
    st.markdown(
    f'<p style="color:white; background-color:#28a745; padding:10px; border-radius:5px; font-size:16px; font-weight:bold;">🏥 Predicted Prescription: {predicted_drug}</p>', 
    unsafe_allow_html=True
)

