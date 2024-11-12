Parkinson's Disease Prediction System

Overview
This project is focused on developing a machine learning model to predict Parkinson’s Disease based on various patient features. The model uses a dataset containing demographic, lifestyle, clinical, and medical history information to determine whether a patient is at risk of developing Parkinson’s Disease.

Key Features:
User Input: The web application built using Streamlit allows users to input personal data such as age, gender, blood pressure levels, family history, and more.

Prediction: The model predicts the likelihood of Parkinson’s Disease with a probability score and a classification of ‘Yes’ or ‘No’ based on the provided data.

Streamlit Interface: An easy-to-use interface where users can input data, view predictions, and receive a probability score of their health status.

Model: The underlying machine learning model (Random Forest, Support Vector Machine, etc.) is trained on the Parkinson's Disease dataset, and the best model is deployed for prediction.

How It Works:

Data Collection: The model uses a dataset containing features such as age, sex, blood pressure, medical history (e.g., hypertension, diabetes), cognitive assessments, and symptoms (e.g., tremor, rigidity).

Feature Engineering & Model Training: The relevant features are extracted, processed, and used to train a classification model. We employ a machine learning pipeline that preprocesses data and makes predictions.

Prediction Logic: When a user inputs their data, the model processes it, provides a classification (whether the patient has Parkinson's or not), and outputs a prediction probability (e.g., 49% chance of having Parkinson’s).

Streamlit UI: An interactive and visually appealing web interface allows users to input their data in real-time, view predictions, and get suggestions based on the predicted probability.

Technologies Used:

Python: For building the machine learning pipeline and web interface.

Streamlit: To create the user-friendly web app for real-time data input and predictions.

Scikit-learn: For building and training the machine learning model.

Joblib: To save and load the trained model.

Pandas & Numpy: For data manipulation and analysis.

Installation:

Clone the repository:

git clone https://github.com/yourusername/parkinsons-disease-prediction.git

Navigate to the project folder:

cd parkinsons-disease-prediction

Create and activate a virtual environment:

python3 -m venv venv

source venv/bin/activate  # On Windows use venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

Future Enhancements:

Adding more features to improve prediction accuracy (e.g., genetic factors).

Deploying the app on a cloud platform like Heroku or AWS for easier access.
