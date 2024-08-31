# Hospital Readmission Prediction Project

This project aims to predict hospital readmissions using machine learning models based on patient data. The goal is to help healthcare providers identify high-risk patients and take proactive measures to prevent readmissions.

# Project Overview

Hospital readmissions are one of the costliest challenges facing healthcare systems. This project builds a predictive model that accurately classifies patients as readmitted or not, based on various features such as age, time in the hospital, number of lab procedures, and more.

# Dataset

The dataset used in this project contains patient records with the following features:

- Age
- Time in hospital
- Number of lab procedures
- Number of medications
- Number of prior outpatient visits
- Number of prior inpatient visits
- Number of prior emergency visits
- Medical specialty
- Diagnosis codes
- Glucose test results
- A1C test results
- Changes in medication
- Diabetes medication
- Readmission status

# Features

Key features used for the prediction:

- Age, time in hospital, lab procedures, etc.
- Diagnosis and treatment information
- Historical data on patient visits

# Model

The model is developed using Python libraries such as T Scikit-Learn, and NumPy, Matplotlib. We employed  machine learning algorithm - Balanced Random Forest to predict readmissions.

# Website and Model Integration

The project includes a web application built using Flask to interact with the predictive model. The website allows users to:

1. Input Patient Data: Users can enter patient information through a user-friendly interface.
2. Predict Readmission: The website sends the input data to the machine learning model, which processes the data and returns a prediction on whether the patient is likely to be readmitted.
3. Display Results: The prediction results are displayed on the website, along with additional information or recommendations.

The web application seamlessly integrates with the machine learning model, ensuring that predictions are fast and reliable. The integration allows healthcare professionals to use the tool without needing extensive technical knowledge.

# Web Application Pages

The web application consists of several key pages:

1. Home Page: Provides an overview of the hospital readmission prediction project, its goals, and its potential impact on healthcare.

2. Data Input of Patient Details Page: Allows users to input patient details such as age, medical history, number of prior visits, and other relevant features. This data is used by the predictive model to determine the likelihood of readmission.

3. Readmission Prediction Page: Displays the prediction results based on the input patient data. It shows whether the patient is likely to be readmitted and provides a confidence score.

4. Model Insight Page: Offers insights into the machine learning model, including how it was trained, which features are most important for predicting readmissions, and the overall performance metrics (accuracy, precision, recall, etc.).

5. About Page: Provides information about the project, the mission and the vision.
