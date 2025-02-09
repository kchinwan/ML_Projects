# 🏡 Real Estate Price Prediction (India Gurgugram City)

## 📌 Overview
This project aims to predict **real estate prices** based on various property features such as bedrooms, bathrooms, area size, floor count, ratings, and more. The dataset consists of real estate listings from an Indian city. The model is built using **Linear Regression** and evaluates price predictions with key metrics like **R-squared, Adjusted R-squared, and Mean Squared Error (MSE).**

## 🔥 Key Features
- **Data Preprocessing:** Cleaned dataset by handling missing values, removing duplicates, and transforming categorical variables.
- **Exploratory Data Analysis (EDA):** Visualized trends, distributions, and correlations between features and target variable.
- **Feature Engineering:** Identified relevant predictors, handled multicollinearity using **Variance Inflation Factor (VIF)**, and scaled data.
- **Model Building:** Used **Linear Regression** to predict property prices.
- **Model Evaluation:** Analyzed model performance using **MSE, R-squared, and Adjusted R-squared**.

---

## 📊 Dataset
### **Features Used:**
| Feature                | Description |
|------------------------|-------------|
| `bedRoom`              | Number of bedrooms in the property |
| `bathroom`             | Number of bathrooms in the property |
| `noOfFloor`            | Total floors in the building |
| `price` (Target)       | Price of the property (in INR) |
| `area_value`           | Area size in square feet |
| `area_type`            | Type of area (e.g., carpet area, built-up area) |
| `balcony_count`        | Number of balconies |
| `agePossession_numeric` | Age of the property |
| `Environment_Rating`   | Rating for environmental factors |
| `Lifestyle_Rating`     | Rating for lifestyle factors |
| `Safety_Rating`        | Safety rating of the locality |
| `Connectivity_Rating`  | Connectivity score of the area |

---

## 📌 Project Structure  ....
House Price Prediction Using Machine Learning
Objective:

To build a machine learning model that accurately predicts house prices based on various features such as location, size, and other property attributes.

Key Steps:

Data Preprocessing & Cleaning:
Handled missing values, outliers, and categorical variables.
Scaled and transformed data for better model performance.
Model Selection & Comparison:
Implemented multiple models:
Linear Regression
Lasso Regression
Ridge Regression
Decision Tree
Random Forest
Compared models using R² score and Mean Squared Error (MSE).
Random Forest performed best with the highest R² score and lowest MSE.
Model Tuning & Deployment:
Tuned hyperparameters of the Random Forest model to improve accuracy.
Saved the trained model as a .pkl file.
Built a Streamlit web app (app.py) to provide an interactive interface for users to predict house prices.
Outcome & Learning:

Gained hands-on experience in machine learning, model evaluation, and deployment.
Successfully created a user-friendly web app for real-time house price prediction.

