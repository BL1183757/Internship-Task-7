# Internship-Task-7

## Overview

This notebook demonstrates the end-to-end process of building a Support Vector Machine (SVM) classification model using scikit-learn pipelines. It covers data preprocessing, model training, evaluation, and visualization.

## Key Steps

### Data Loading & Exploration

  Reads dataset into a Pandas DataFrame.

  Performs basic EDA using pandas, numpy, and seaborn.

### Data Preprocessing

  Identifies numerical and categorical features.

  Uses ColumnTransformer with:

   StandardScaler for numerical columns.

   OrdinalEncoder for categorical columns.

### Model Building

   Implements an SVC (Support Vector Classifier) inside a Pipeline.

   Trains the model on the training split using train_test_split.

### Model Evaluation

   Generates Confusion Matrix and Classification Report.

   Calculates regression-style metrics (MAE, MSE, R²) for performance insight.

   Uses cross-validation for robustness check.

### Model Saving

   Saves the trained pipeline using joblib.

## Dependencies
    pip install pandas numpy matplotlib seaborn scikit-learn joblib
