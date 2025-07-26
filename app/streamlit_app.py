
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("model/furniture_model.pkl")

# UI Title
st.title("🪑 E-Commerce Furniture Sales Predictor")
st.markdown("Predict **sales volume** for your furniture based on marketing, traffic, and ratings.")

# Sidebar inputs
st.sidebar.header("Input Features")
TV = st.sidebar.slider("TV Advertising ($)", 0, 300, 150)
Radio = st.sidebar.slider("Radio Advertising ($)", 0, 100, 50)
Social_Media = st.sidebar.slider("Social Media Advertising ($)", 0, 300, 100)
Influencer = st.sidebar.selectbox("Influencer Type", ["Nano", "Micro", "Macro", "Mega"])
Traffic = st.sidebar.slider("Website Traffic (visits)", 0, 100000, 50000)
Product_Views = st.sidebar.slider("Product Views", 0, 10000, 5000)
Add_to_Cart = st.sidebar.slider("Add to Cart", 0, 5000, 1000)
Rating = st.sidebar.slider("Average Rating", 0.0, 5.0, 4.0, 0.1)

# Categorical encoding
influencer_map = {"Nano": 0, "Micro": 1, "Macro": 2, "Mega": 3}
Influencer_encoded = influencer_map[Influencer]

# Prepare input
input_data = pd.DataFrame([[TV, Radio, Social_Media, Influencer_encoded, Traffic, Product_Views, Add_to_Cart, Rating]],
                          columns=["TV", "Radio", "Social_Media", "Influencer", "Traffic", "Product_Views", "Add_to_Cart", "Rating"])

# Predict
if st.button("Predict Sales"):
    prediction = model.predict(input_data)[0]
    st.success(f"📦 Predicted Sales Volume: **{int(prediction):,} units**")
