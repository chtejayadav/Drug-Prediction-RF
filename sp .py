import streamlit as st
import pandas as pd
import pickle
import numpy as np
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ✅ Set Page Config (Must be the first Streamlit command)
st.set_page_config(page_title="Patient Drug Prediction", layout="wide")

# ✅ Function to Set Background Image (Works with URLs)
def set_background(image_url):
    background_style = f"""
    <style>
    .stApp {{
        background: url("{image_url}") no-repeat center center fixed;
        background-size: cover;
    }}
    h1, h2, h3, h4, h5, h6, label, span {{
        color: white !important; /* Ensures white text */
    }}
    section[data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.7) !important; /* Darkened Sidebar */
        color: white !important;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# ✅ Background Image (Replace with Your GitHub Raw URL)
bg_url = "https://raw.githubusercontent.com/your-username/your-repo/main/bgg.gif"
set_background(bg_url)

# ✅ Load the trained model
@st.cache_data
def load_model():
    with open("rf_model.pkl", "rb") as f:
        return pickle.load(f)

# ✅ Train and Save Model (if not already trained)
def train_model():
    file_path = "sp.csv"
    df = pd.read_csv(file_path)
    df = df.drop(columns=["Patient_ID"])  

    # Encode categorical variables
    label_encoders = {}
    for col in ["Gender", "Prescribed_Drug"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Define features and target
    X = df.drop(columns=["Prescribed_Drug"])
    y = df["Prescribed_Drug"]

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save Model & Encoders
    with open("rf_model.pkl", "wb") as f:
        pickle.dump((model, label_encoders), f)

# ✅ Train Model (Only if Not Already Trained)
train_model()

# ✅ Load Model & Encoders
model, label_encoders = load_model()

# ✅ Load Feature Names from CSV
df = pd.read_csv("sp.csv").drop(columns=["Patient_ID"])
feature_names = df.drop(columns=["Prescribed_Drug"]).columns

# ✅ Streamlit UI
st.title("💊 Patient Drug Prescription Prediction")
st.markdown("### Enter patient details to predict the prescribed drug.")

# ✅ Sidebar - About Section
with st.sidebar:
    st.header("🔍 About the App")
    st.markdown("""
    - **Purpose**: Predicts the prescribed drug based on patient details.  
    - **Technology Used**: Streamlit, Machine Learning (Random Forest), Pandas.  
    - **Input Features**:  
      - Age, Gender  
      - Blood Sugar (Fasting & Post-Meal)  
      - HbA1c, BMI  
      - Blood Pressure (Systolic & Diastolic)  
      - Cholesterol Level  
    - **Prediction Output**: Displays the most likely prescribed drug.  
    """)

# ✅ Form for User Input
with st.form("patient_form"):
    st.subheader("📝 Patient Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        fasting_blood_sugar = st.number_input("Fasting Blood Sugar", min_value=50, max_value=300, value=100)

    with col2:
        post_meal_blood_sugar = st.number_input("Post-Meal Blood Sugar", min_value=50, max_value=500, value=140)
        hba1c = st.number_input("HbA1c", min_value=3.0, max_value=15.0, step=0.1, value=5.5)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1, value=25.0)

    with col3:
        blood_pressure_sys = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
        blood_pressure_dia = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=130, value=80)
        cholesterol_level = st.number_input("Cholesterol Level", min_value=100, max_value=300, value=200)

    submit_button = st.form_submit_button("💊 Predict Prescription")

# ✅ Perform Prediction
if submit_button:
    # Convert input to DataFrame
    input_data = pd.DataFrame([[age, gender, fasting_blood_sugar, post_meal_blood_sugar, hba1c, bmi,
                                blood_pressure_sys, blood_pressure_dia, cholesterol_level]],
                              columns=["Age", "Gender", "Fasting_Blood_Sugar", "Post_Meal_Blood_Sugar", "HbA1c",
                                       "BMI", "Blood_Pressure_Sys", "Blood_Pressure_Dia", "Cholesterol_Level"])

    # Encode Gender
    input_data["Gender"] = label_encoders["Gender"].transform([gender])

    # Make Prediction
    prediction = model.predict(input_data)[0]
    predicted_drug = label_encoders["Prescribed_Drug"].inverse_transform([prediction])[0]

    # ✅ Display Result
    st.markdown(
        f'<p style="color:white; background-color:#28a745; padding:10px; border-radius:5px; font-size:18px; font-weight:bold;">🏥 Predicted Prescription: {predicted_drug}</p>',
        unsafe_allow_html=True
    )
