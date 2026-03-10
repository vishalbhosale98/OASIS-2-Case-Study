# Alzheimer's Disease Risk Prediction

This project builds a machine learning model to predict the risk of Alzheimer's disease using clinical and MRI-based features from the OASIS longitudinal dataset.

The pipeline includes data preprocessing, feature engineering, feature selection, model training, and deployment using FastAPI.

---

## Project Workflow

## Project Workflow

1. Data Cleaning & Preprocessing
2. Feature Engineering
3. Outlier Treatment
4. Feature Selection (Correlation Analysis and VIF)
5. Group-based Train/Test Split (to avoid subject leakage)
6. Model Training

   * Logistic Regression
   * Decision Tree
   * Random Forest
   * AdaBoost
7. Model Evaluation
8. Feature Importance & SHAP Explainability
9. Model Deployment using FastAPI 


---

## Installation

Clone the repository and install dependencies:

pip install -r requirements.txt

---

## Model Deployment

The trained Logistic Regression model is saved and served through a FastAPI REST API for inference.

Run the API:

uvicorn app:app --reload

API will start at:

http://127.0.0.1:8000

Interactive documentation:

http://127.0.0.1:8000/docs

---

## Example API Request

POST /predict

{
  "SES": 2,
  "time_since_first_visit": 2,
  "nWBV_change": -0.01,
  "brain_atrophy_rate": -0.005,
  "visit_count": 3
}

Response

{
  "prediction": 1,
  "probability_of_dementia": 0.81
}

---

## Tech Stack

Python  
Scikit-learn  
SHAP  
FastAPI  
Pandas  
Joblib  

---

## Author

Vishal Bhosale