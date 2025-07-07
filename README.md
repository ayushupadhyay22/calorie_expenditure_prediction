# Calorie Expenditure Prediction

This project aims to predict calorie expenditure using Kaggle dataset https://www.kaggle.com/competitions/playground-series-s5e5/overview.

## Project Overview
- **Goal:** Build and evaluate machine learning models to predict the number of calories burned.
- **Tech Stack:** Python, scikit-learn, pandas, matplotlib, seaborn, Jupyter Notebook
- **Models Used:** Linear Regression, Decision Tree Regressor, Random Forest Regressor
- **Evaluation:** Root Mean Squared Error (RMSE), Cross-validation

## Dataset
The dataset is provided by Kaggle and includes the following files in the `data/` directory:
- `train.csv`: Training data with features and target (`Calories`)
- `test.csv`: Test data for final predictions

**Features include:**
- Age, Height, Weight, Duration, Heart_Rate, Body_Temp, Sex
- **Target:** Calories (number of calories burned)

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/ayushupadhyay22/calorie_expenditure_prediction
   cd calorie_expenditure_prediction
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is missing, install: pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter)*
4. **Download the dataset:**
   - Use the Kaggle CLI to download the competition data into the `data/` folder.

## Usage
- **Data Exploration:**
  - Use the provided Jupyter notebooks (e.g., `exploration.ipynb`) to explore and visualize the data.
- **Data Preparation:**
  - Feature engineering, encoding categorical variables, and scaling are handled in the pipeline.
- **Model Training & Evaluation:**
  - Train and evaluate models using the provided notebooks (e.g., `calorie_prediction.ipynb`).
  - Includes cross-validation and actual vs. predicted plots for model comparison.

## Model Evaluation
- **Metrics:**
  - Root Mean Squared Error (RMSE)
  - Standard deviation of prediction errors
  - 10-fold cross-validation for robust evaluation
- **Visualization:**
  - Actual vs. Predicted scatter plots for each model

## Results
- Random Forest Regressor generally provides the best performance, balancing accuracy and generalization.
- Linear Regression is simple but may underfit.
- Decision Tree can overfit the training data.