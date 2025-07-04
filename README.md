# Employee-Promotion 
A machine learning project to predict employee promotions using classification algorithms and data preprocessing techniques.

## Overview

This project helps HR teams predict whether an employee is eligible for promotion based on factors like demographics, performance ratings, training scores, and other relevant attributes.

## Dataset

The dataset contains employee information including:
- Demographics (age, gender, education, region)
- Performance metrics (previous year rating, training scores)
- Experience (length of service, department)
- Achievements (awards won, number of trainings)
- **Target**: `is_promoted` (1/0)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/employee-promotion-prediction.git
cd employee-promotion-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place `employee_promotion.csv` in the project root directory

## Usage

Run the analysis:
```bash
python "Employee Promotion_project.py"
```

## Key Features

- **Data Exploration**: Comprehensive EDA with visualizations
- **Machine Learning Models**: Random Forest, AdaBoost, Gradient Boosting, XGBoost
- **Data Balancing**: SMOTE oversampling and undersampling
- **Hyperparameter Tuning**: Optimized model performance
- **Evaluation**: F1-score, accuracy, precision, recall metrics

## Results

- **Best Model**: Gradient Boosting with undersampled data
- **Key Features**: Average training score, previous year rating, length of service
- **Evaluation**: Comprehensive performance metrics and model comparison

## Requirements

See `requirements.txt` for dependencies.
