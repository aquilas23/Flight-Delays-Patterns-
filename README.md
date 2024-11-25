# Flight Delay Prediction Project
## Overview
This project focuses on analyzing and predicting flight delays using machine learning and deep learning models. It includes a thorough exploration of the dataset, preprocessing steps, feature engineering, and the application of various regression and classification models. The project also evaluates model performance using appropriate metrics and visualizations.

## Dataset
## Source: Flight Delay and Cancellation Dataset (2019-2023)
## Size: Approximately 3 million rows with 32 features
## Key Features:
DEP_DELAY: Departure delay (minutes)
ARR_DELAY: Arrival delay (minutes)
CANCELLED: Flight cancellation indicator (0 = Not cancelled, 1 = Cancelled)
CANCELLATION_CODE: Reason for cancellation
ORIGIN: Origin airport code
DEST: Destination airport code
Weather-related delay causes (DELAY_DUE_WEATHER, DELAY_DUE_CARRIER, etc.)

## Objectives
Classification: Predict whether a flight is cancelled or not (CANCELLED as target).
Regression: Predict arrival delay (ARR_DELAY) based on departure delay and other features.

## Preprocessing
Missing Value Handling:
Imputed DEP_DELAY and ARR_DELAY with the median.
Filled delay cause columns with 0 for missing values.
Feature Engineering:
Encoded CANCELLED as 0 (On-time/Early) and 1 (Cancelled).
Created additional temporal features like MONTH.

## Models
Classification Models
Logistic Regression: Baseline model for binary classification.
Decision Tree Classifier: Captures non-linear relationships in the data.
Random Forest Classifier: Ensemble model for robust performance.
Evaluation Metrics: Confusion Matrix, Precision, Recall, F1-score, Accuracy.
Regression Models
Linear Regression: Predicts ARR_DELAY based on DEP_DELAY.
Decision Tree Regressor: Non-linear regression model.
Random Forest Regressor: Ensemble regression model.
Artificial Neural Network (ANN): Multi-layer perceptron model for regression.
Long Short-Term Memory (LSTM): Sequence model for handling temporal dependencies.
Evaluation Metrics: Mean Squared Error (MSE), R-squared (R²).
Visualizations
Confusion Matrices: Performance evaluation for classification models.
Actual vs Predicted Plots: Regression performance analysis.
Training vs Validation Loss: Loss curves for ANN and LSTM models.
Model Comparison: Bar plots for R² and MSE metrics.

## Requirements
Python 3.8+
## Libraries
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow or keras
Install dependencies using:

## Results
Classification
Logistic Regression achieved 99% accuracy, but due to class imbalance, F1-score for the minority class is low.
Random Forest Classifier showed better handling of imbalanced data with higher recall for cancelled flights.
Regression
ANN achieved the best performance with an MSE of 203.66 and R² of 0.91.
LSTM performed comparably with MSE of 221.40 and R² of 0.90.
