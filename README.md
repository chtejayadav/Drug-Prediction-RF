# 💊 Patient Drug Prescription Prediction

## Overview
This project is a **Machine Learning-based web application** that predicts the prescribed drug for a patient based on input health parameters. The model uses **Random Forest** classification trained on medical data. The application is built using **Python, Streamlit, Pandas, Matplotlib, Seaborn, and Pickle for model handling.** The interface is designed for ease of use with a clean and intuitive layout.

---

## 🚀 Features

- **User-friendly Interface:** Built with **Streamlit** for seamless interaction.
- **Accurate Predictions:** Uses a **pre-trained Random Forest model** for drug prediction.
- **Custom Styling:** Dark theme UI with a background image.
- **Data Input Form:** User inputs patient details via an interactive form.
- **Sidebar Navigation:** Includes an *About* section and additional settings.

## 🖥️ Tech Stack

- **Languages & Tools:** Python, Streamlit, Pandas, Matplotlib, Seaborn, Pickle
- **Machine Learning Model:** **Random Forest**
- **Background Customization:** Styled using **CSS** and inline **HTML** in Streamlit

---

## 🏗 Project Structure
```
📂 patient_drug_prediction
│── app.py              # Streamlit web app script
│── requirements.txt      # Required Python dependencies
│── model.pkl             # Trained Machine Learning model
│── data
│   ├── sp.csv            # Dataset used for model training
│── assets
│   ├── bgg.gif          # Background GIF for the UI
│── LICENSE               # License information
│── README.md             # Project Documentation
```

## 🛠️ Installation Guide

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-repo.git
   cd patient_drug_prediction
   ```

2. **Create and Activate Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

1. **Start the Application:**
   ```bash
   streamlit run app.py
   ```
2. Enter the patient details, such as:
   - Gender
   - Fasting Blood Sugar
   - Post Meal Blood Sugar
   - HBA1C level
   - Blood Pressure
   - Cholesterol Level
3. Click on **Predict** to get the prescribed drug for the patient.
4. The results will be displayed along with the background visualization.

---

## 📂 Dataset Information

The model is trained on a dataset named `sp.csv`, containing the following patient information:

- **Gender** *(Male/Female)*
- **Fasting Blood Sugar**
- **Post Meal Blood Sugar**
- **HBA1C Levels**
- **Cholesterol Level**
- **Prescribed Drug**

The dataset is preprocessed using **Pandas** and categorical variables are encoded using `LabelEncoder`.

---

## 🚀 How to Run the Project Locally

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd patient_drug_prediction
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Open your browser and go to `http://localhost:8501/` to use the app.

---
## 💾 Model Training & Updating

To update the model with new data:
1. Add new patient data to the `sp.csv` file.
2. Run `app.py` to train the **Random Forest Model** again and save the updated version.

## 📜 License

This project is licensed under the **MIT License**.

## 📝 Author

Developed by: **CH TEJA YADAV**  
📧 Email: tejayadavch@gmail.com  
💻 GitHub: [YourGitHubProfile](https://github.com/yourusername)  
📢 Feel free to reach out for any questions or contributions!

---

### 📜 Acknowledgments
- **Streamlit** for the interactive UI.
- **Pandas & Seaborn** for data manipulation and visualization.
- **Scikit-Learn** for Machine Learning implementation.

## 🔧 Future Enhancements

- Deploying the model using cloud platforms like **Heroku**.
- Improving model accuracy with hyperparameter tuning.
- Extending the dataset for better generalization.
- Adding more patient features for enhanced accuracy.
- Implementing authentication for secure data input.

## 💡 License
This project is licensed under the **MIT License** - feel free to use and modify it as needed.

---

If you find this project useful, feel free to ⭐ the repository and contribute!

