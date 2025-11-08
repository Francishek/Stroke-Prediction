# DS.v2.5.3.2.5

# Stroke Prediction Model

## Introduction

**Context**

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. 

This dataset contains relevant patient information in each row, including parameters such as gender, age, various diseases, and smoking status. 

It is designed to be used for building a machine learning model that can predict a patient's likelihood of experiencing a stroke. The ability to accurately determine which patients have a high stroke risk will empower doctors to provide timely advice and guidance to both patients and their families on how to act in case of an emergency.


**Goals:**

        Reduce preventable stroke deaths by identifying high-risk patients for proactive care.

        Improve clinical decision-making with interpretable risk predictions.

**Objectives:**


        Develop a model with ≥70% recall (minimize missed strokes) while keeping precision ≥15%.

        Ensure generalizability by validating on test set (ROC-AUC ≥0.8 and PR-AUC ≥0.2).       
        
        Identify top 3 clinical risk factors driving predictions using SHAP values.

        Deploy the model.


**Dataset Overview**

The dataset contains 5110 patient observations. It includes 11 independent variables (6 numerical, 5 categorical) and 1 target variable (Stroke):

**Feature descriptions:**
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood (mg/dL)
10) bmi: body mass index (kg/m2)
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown" ("Unknown" in smoking_status means that the information is unavailable for this patient)
12) stroke: 1 if the patient had a stroke or 0 if not.

## Data Source

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data).

## Setup

1. Clone the repo:

   Link [GitHub](https://github.com/Francishek/Stroke-Prediction) or
   ```bash
   git clone https://github.com/Francishek/Spaceship-Titanic
   cd project-root
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## To use the model for predictions:

        1: Check requirements.
        
        2: Clone the Repository /bash git clone <repo-url> /bash cd <repo-folder-name>

        3: Launch the App /bash streamlit run app.py {Streamlit will launch the app in your browser (usually at http://localhost:8501)}

        4: a) Upload Your Data: use the "Browse files" or drag-and-drop box to upload your .csv or .xlsx
           
           b) Imput information manually

        5: View Predictions: the app will show a table of predictions (probabilities and Stroke: 0/1).

## Jupyter Notebook Structure:

## 1. Introduction

## 2. Exploratory Data Analysis (EDA)

### A. Data loading & Initial checks

### B. Univariate Analysis

### C. Multivariate Analysis

## 3. Statistical Inference

## 4. Machine Learning Modeling

### A. Feature Engineering

### B. Data Preparation

### C. Pipeline Preprocessing 

### D. Model Selection 

### E. Hyperparameter Tuning

### E. Ensembling

### F. Ensemble evaluation on Test set

## 5. Conclusion

The final ensemble model, combining Logistic Regression, XGBoost, and LightGBM, met all performance objectives and outperformed individual classifier in F1 score and ROC-AUC. 

The model prioritizes high recall—critical for stroke detection—while maintaining precision above the required threshold. 
        
Key predictors of stroke risk include age, average glucose level, and BMI. 
        
Extensive validation and error analysis confirmed the model's robustness and clinical relevance.
