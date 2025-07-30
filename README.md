# 🚢 Titanic Survival Predictor 🚢

This project is a Machine Learning-based web application that predicts whether a passenger would have survived the Titanic disaster based on input features like age, gender, class, fare, etc.

## 🖼️ Screenshots

### 🏠 Home Page
![Home Page](static/screenshots/homepage.png) <!-- Replace with your actual image path -->

### 📊 Prediction Result
![Prediction Output](static/screenshots/prediction.png)


## 📌 Project Description

The Titanic Survival Predictor uses historical passenger data to train a machine learning model to classify survival outcomes. This is a classic binary classification problem using supervised learning.

## 🧠 Features Used

- Pclass (Passenger Class)
- Sex (Gender)
- Age
- SibSp (Siblings/Spouses Aboard)
- Parch (Parents/Children Aboard)
- Fare
- Embarked (Port of Embarkation)

## ⚙️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn (Logistic Regression, Random Forest, etc.)
- Matplotlib, Seaborn (for data visualization)
- Flask (for deployment)
- HTML, CSS (for frontend)

## 🧪 ML Workflow

1. Data Cleaning & Preprocessing
2. Feature Engineering
3. Handling Missing Values
4. Model Training (Logistic Regression / Random Forest)
5. Model Evaluation (Accuracy, Confusion Matrix)
6. Deployment with Flask

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/seelamdivya23/Titanic-survival-predictor.git
cd Titanic-survival-predictor

# (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
