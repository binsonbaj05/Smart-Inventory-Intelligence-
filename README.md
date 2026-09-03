# Smart-Inventory-Intelligence-
Project Description

This project develops a machine-learning based Demand Forecasting workflow for predicting sales quantity using product, store, pricing, promotion, seasonality, external-factor, and customer-segment information.

The notebook combines three datasets — demand forecasting, inventory monitoring, and pricing optimization — using Product ID and Store ID. It performs data-quality checking, exploratory data analysis, feature engineering, categorical encoding, model training, model comparison, cross-validation, and hyperparameter tuning.

Objectives

Analyze historical sales and demand patterns.

Identify missing values and duplicate records.

Combine demand, inventory, and pricing information.

Extract month, quarter, and day-of-week features from dates.

Encode categorical variables for machine learning.

Compare multiple regression models.

Evaluate models using MAE, RMSE, R2, and cross-validation R2.

Tune a Random Forest model using GridSearchCV.

Build a workflow for demand prediction.

Datasets

The notebook loads:

demand_forecasting.csv

inventory_monitoring.csv

pricing_optimization.csv

The demand dataset contains 10,000 rows and 10 columns in the recorded notebook run. It includes Product ID, Date, Store ID, Sales Quantity, Price, Promotions, Seasonality Factors, External Factors, Demand Trend, and Customer Segments.

Workflow

Load Datasets
      ↓
Data Quality Check
      ↓
Exploratory Data Analysis
      ↓
Merge Demand + Inventory + Pricing
      ↓
Date Feature Engineering
      ↓
Categorical Encoding
      ↓
Missing-Value Handling
      ↓
Feature Selection
      ↓
Train/Test Split
      ↓
Train Regression Models
      ↓
Evaluate Models
      ↓
Cross-Validation
      ↓
Hyperparameter Tuning
      ↓
Demand Prediction

Data Quality

The recorded notebook run found missing values in Seasonality Factors and External Factors in the demand dataset and reported no duplicates. Inventory and pricing datasets were reported with no missing values and no duplicates.

EDA

The notebook analyzes:

Sales quantity distribution

Sales by seasonality

Sales by promotion status

Price vs. sales correlation

The EDA figure is saved as outputs/eda_demand.png.

Preprocessing and Features

The datasets are merged on Product ID and Store ID. The Date field is converted to datetime and the following features are created:

Month

Quarter

DayOfWeek

Categorical variables are encoded with LabelEncoder and numerical missing values are filled using median values.

Forecasting features:

Price
Promotions
Seasonality Factors
External Factors
Customer Segments
Month
Quarter
DayOfWeek

Target:

Sales Quantity

Machine Learning Models

The notebook compares:

Linear Regression

Random Forest Regressor

Gradient Boosting Regressor

Evaluation Metrics

MAE: Mean Absolute Error; lower is better.

RMSE: Root Mean Squared Error; lower is better.

R2: R-squared; higher is generally better.

CV_R2: mean 5-fold cross-validation R2.

Recorded Model Results

Model

MAE

RMSE

R2

CV R2

Linear Regression

125.63

145.21

-0.0017

-0.0007

Gradient Boosting

126.35

146.31

-0.0170

-0.0138

Random Forest

126.65

146.81

-0.0239

-0.0213

In the recorded run, Linear Regression had the best R2/CV R2 and the lowest MAE/RMSE among the three compared models.

The notebook also tunes Random Forest with GridSearchCV. Recorded best parameters:

n_estimators = 200
max_depth = 10
min_samples_split = 5

Recorded tuned Random Forest:

Best CV R2: -0.0168
Test R2:     -0.0158
Test MAE:    126.26

The negative R2 values indicate that the current feature set does not explain sales quantity well in this recorded run. This provides a clear area for future improvement rather than evidence of a strong forecasting model.

Technologies

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Joblib

Jupyter Notebook

Suggested Project Structure

Smart_Inventory_Intelligence/
├── data/
│   ├── demand_forecasting.csv
│   ├── inventory_monitoring.csv
│   └── pricing_optimization.csv
├── notebooks/
│   └── Demand_Forecasting.ipynb
├── models/
├── outputs/
│   └── eda_demand.png
└── README.md

Installation

pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter

Run

Start Jupyter:

jupyter notebook

Open the Demand Forecasting notebook and run the cells from top to bottom. Make sure the three CSV files are in the data folder.

Future Improvements

Use time-series train/test splitting.

Add lag and rolling-demand features.

Add richer historical demand information.

Improve handling of missing data after dataset merging.

Add automated feature selection.

Test advanced forecasting approaches.

Add forecasting charts and prediction intervals.

Save the final preprocessing pipeline and model.

Integrate the forecasting model into the Automated Data Science System dashboard.

Purpose

This project demonstrates an end-to-end machine-learning workflow for demand forecasting, from data integration and EDA to regression model comparison, evaluation, and hyperparameter tuning.
