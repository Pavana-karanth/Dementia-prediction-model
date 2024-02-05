# Dementia-prediction-model
This repository contains code for a Dementia Prediction App, including both the training of machine learning models and a Streamlit app for making predictions.

📌**Table of Contents:**
- Introduction
- Files
- Model Training
- Model Prediction
- Streamlit App
- License

📌**Files**
- Dementia_model_training.ipynb: Jupyter Notebook containing the code for training machine learning models using the provided dataset.
- Dementia_prediction_model.py: Python script for loading the trained models and creating a Streamlit app for making predictions.
  
_Other diles created while coding in notebook:_
- _X.joblib: Joblib file containing the feature matrix X used during training.
- best_decision_tree_model.joblib: Joblib file containing the best-trained Decision Tree model.
- best_random_forest_model.joblib: Joblib file containing the best-trained Random Forest model.
  
 _Dataset imported from kaggle:_
- health_dementia_data.csv: CSV file containing the dataset used for training.
  <https://www.kaggle.com/datasets/gilbertmilton20/dementia-patient-characteristics-dataset>

📌**Model Training**

To train the machine learning models, run the Jupyter Notebook _Dementia_model_training.ipynb_. This notebook covers loading the dataset, exploring data, training models, and hyperparameter tuning.

📌**Model Prediction**

To make predictions using the trained models, run the Python script _Dementia_prediction_model.py_. This script loads the models, takes user input through a Streamlit app, and displays predictions.

📌**Streamlit App**

The Streamlit app simplifies user interaction with the trained models. Users can input their health-related information, and the app will provide predictions based on the trained models.

📌**License**

This project is licensed under the MIT License - see the LICENSE file for details.
