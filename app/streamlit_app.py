import streamlit as st
import pandas as pd
import joblib
import os

# Load model from models/ folder
model_path = os.path.join("models", "furniture_model.pkl")
model = joblib.load(model_path)

st.title("🛋️ E-commerce Furniture Sales Predictor")

st.markdown("### Enter Marketing and Traffic Data")

# Input fields
tv = st.number_input("TV Advertising Budget", min_value=0)
radio = st.number_input("Radio Advertising Budget", min_value=0)
social = st.number_input("Social Media Budget", min_value=0)
influencer = st.selectbox("Influencer Category", [0, 1, 2, 3])
traffic = st.number_input("Website Traffic", min_value=0)
views = st.number_input("Product Views", min_value=0)
cart = st.number_input("Add to Cart Count", min_value=0)
rating = st.slider("Average Product Rating", 1.0, 5.0, 4.0, step=0.1)

# Prediction
if st.button("Predict Sales"):
    input_df = pd.DataFrame([[tv, radio, social, influencer, traffic, views, cart, rating]],
                            columns=["TV", "Radio", "Social_Media", "Influencer", "Traffic", "Product_Views", "Add_to_Cart", "Rating"])
    prediction = model.predict(input_df)[0]
    st.success(f"📈 Predicted Sales: ₹{int(prediction):,}")
