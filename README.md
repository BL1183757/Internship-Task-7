# Internship-Task-7

## Breast Cancer Diagnosis Using SVM

This project presents a machine learning pipeline to classify breast cancer tumors as benign or malignant using the **Support Vector Machine (SVM)** classifier. It makes use of the popular Breast Cancer Wisconsin (Diagnostic) Dataset and leverages data preprocessing, feature scaling, and pipeline automation for robust results.

## Project Overview

- **Dataset:** Breast Cancer Wisconsin (Diagnostic), consisting of features computed from digitized images of fine needle aspirate (FNA) of breast masses.
- **Goal:** Predict whether a tumor is benign (B) or malignant (M) based on given features.
- **Algorithm:** Support Vector Classifier (SVC) with a linear kernel.
- **Libraries Used:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

## Dataset Description

The dataset contains the following columns:
- **id:** Unique identifier for each tumor sample.
- **diagnosis:** Tumor classification (`M` = malignant, `B` = benign).
- **30 numeric features:** Including radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension (mean, standard error, and worst/largest).
- **Unnamed: 32:** Unused column (NaN).

## Installation

### Install necessary libraries:
    pip install pandas numpy matplotlib seaborn scikit-learn


## Workflow

1. **Data Loading & Exploration**
   - Import the dataset into a Pandas DataFrame.
   - Check for missing values and examine feature distributions.

2. **Data Preprocessing**
   - Remove unnecessary columns and handle missing data.
   - Separate features (`X`) and target labels (`y`).
   - Scale numerical features using `StandardScaler` for better algorithm performance.
   - No categorical features are present apart from the label.

3. **Train-Test Split**
   - Split the dataset into training and testing sets (e.g., 80% train, 20% test).

4. **Pipeline Construction**
   - Build a `scikit-learn` pipeline that combines preprocessing and SVM classifier.

5. **Model Training & Evaluation**
   - Train the model on the training data.
   - Predict on the test data.
   - Evaluate performance using metrics such as accuracy, mean absolute error (MAE), mean squared error (MSE), and R² score.

6. **Visualization**
   - Utilize Matplotlib or Seaborn (e.g., visualize feature importance, data distribution).

## Results

- SVM classifier demonstrates high accuracy and reliable differentiation between benign and malignant tumors.
- Robust processing with pipeline automation ensures reproducibility and scalability of the machine learning workflow.

## How to Run

### Clone this repository
    git clone https://github.com/BL1183757/Internship-Task-7.git
    cd Internship-Task-7

### Install dependencies
    pip install -r requirements.txt

### Run the Notebook
    jupyter notebook Task-7.ipynb

4. Execute all cells sequentially to reproduce the results and visualizations.

## Model Performance Metrics

- **Accuracy Score:** Measures the proportion of correct predictions.
- **MAE & MSE:** Quantify average prediction errors.
- **R² Score:** Indicates how well the model explains the variance in data.

## License

This project is for educational and research purposes. Feel free to use and adapt it as needed.

---

**Author:** *Bhavay Khandelwal*  
**Date:** 15th August 2025




