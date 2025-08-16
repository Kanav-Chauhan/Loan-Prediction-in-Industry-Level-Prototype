
🏦 Loan Status Prediction - [Click for Live Demo](https://kanav-loan-predict.streamlit.app/)
=========================================

Project Overview
----------------
This project predicts whether a loan application will be approved or rejected based on applicant data.
It uses multiple machine learning models, compares their performance, and deploys the best model in a Streamlit web application.

Features
--------
- Multi-Model Comparison: Logistic Regression, Random Forest, Gradient Boosting, SVC, Decision Tree, KNN
- Pipeline Integration: Combines preprocessing (imputation, scaling, encoding) and model in a single pipeline
- Random Input Generator: Quickly generate random loan applicant data for testing
- Prediction with Probability: Shows loan approval probability with a dynamic progress bar
- Performance Visualization: Horizontal bar chart of F1 scores for all models

Preprocessing Steps
------------------
1. Missing Value Imputation
   - Numeric: Median
   - Categorical: Most Frequent
2. Scaling
   - StandardScaler applied to numeric features
3. Encoding
   - One-Hot Encoding for categorical features
4. ColumnTransformer & Pipeline
   - Ensures preprocessing and model are combined and reproducible

Model Evaluation
----------------
- Cross-validation: Stratified K-Fold (n=5)
- Metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Best Model Selection: Automatically identifies the best performing model (highest F1 score) and saves it with joblib

Streamlit App Features
----------------------
- Single Dashboard Layout
  - Compact input boxes side by side
  - Random button to fill inputs with example data
  - Predict button to show loan approval/rejection probability
- Horizontal F1 Score Graph
  - Visual comparison of all models’ performance

How to Run
----------
1. Install requirements:
   
   ```bash
   pip install -r requirements.txt 
2. Run the Streamlit app:
 
   ```bash
   streamlit run app.py
4. Open the link provided in the terminal to interact with the dashboard

Real World Applications
----------------------
- Automating loan approval decisions for banks or financial institutions
- Quickly evaluating applicant risk based on historical data
- Can be extended for credit scoring or risk assessment systems

Project Summary
---------------
- Loaded and preprocessed loan dataset
- Used pipelines for clean and reproducible ML workflow
- Tuned multiple models with RandomizedSearchCV
- Saved best model with its parameters for production
- Built interactive Streamlit dashboard for end users
# Made with ❤️ by Kanav Chauhan
