import streamlit as st
import pickle
import numpy as np
import os

# Load the trained model
print(os.getcwd())
model_path = "../models/xgboost_model.pkl"  # Adjust if using a different model
with open(model_path, "rb") as file:
    model = pickle.load(file)

# App Title
st.title("House Price Prediction App")
st.write("Enter the property details to get a price prediction.")

# Input fields for user
bedroom = st.number_input("Number of Bedrooms", min_value=1, step=1)
bathroom = st.number_input("Number of Bathrooms", min_value=1, step=1)
no_of_floors = st.number_input("Number of Floors", min_value=1, step=1)
area_value = st.number_input("Area Size (sq ft)", min_value=500, step=100)
balcony_count = st.number_input("Number of Balconies", min_value=0, step=1)
age = st.number_input("Age of Property (years)", min_value=0, step=1)
safety = st.slider("Safety Rating", 1, 5, 3)
lifestyle = st.slider("Lifestyle Rating", 1, 5, 3)
environment = st.slider("Environment Rating", 1, 5, 3)
connectivity = st.slider("Connectivity Rating", 1, 5, 3)

# Predict Button
if st.button("Predict Price"):
    # Prepare input data
    input_data = np.array([
        [bedroom, bathroom, no_of_floors, area_value, balcony_count, age, 
         safety, lifestyle, environment, connectivity]
    ])
    
    # Predict the price
    prediction = model.predict(input_data)[0]
    st.success(f" Estimated House Price: ₹{prediction:,.2f} Crore")
