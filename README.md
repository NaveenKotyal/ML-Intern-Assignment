# Hyperspectral Imaging for Mycotoxin Prediction

## Project Description

This project focuses on analyzing hyperspectral imaging data to predict DON (vomitoxin) concentration in corn samples. The dataset contains spectral reflectance values across multiple wavelength bands for different samples, with the goal of building a regression model to estimate mycotoxin levels. The key steps include:

Data Preprocessing: Handling missing values, normalizing spectral data, and visualizing spectral bands.

Exploratory Data Analysis (EDA): Examining target variable distribution, correlations, and spectral trends.

Dimensionality Reduction: Applying PCA to reduce feature dimensions while retaining variance.

Machine Learning Modeling: Training and evaluating Random Forest and XGBoost models for regression.

Model Evaluation: Using MAE, RMSE, and R² score to assess model performance and visualizing predictions.

## Technologies Used

Python

Pandas, NumPy

Scikit-learn

XGBoost

Matplotlib, Seaborn

### 1. Clone the Repository
```bash
git clone https://github.com/NaveenKotyal/ML-Intern-Assignment.git
cd  ML-Intern-Assignment

### 2. Create a Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate 

### 3. Install Dependencies
pip install -r requirements.txt


