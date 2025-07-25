
import streamlit as st
import pandas as pd
import joblib

st.title("🛋️ Furniture Sales Predictor")

model = joblib.load("../models/rf_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

title = st.text_input("Product Title")
price = st.number_input("Price", min_value=0.0)
tag = st.selectbox("Shipping Tag", ["Free shipping", "+Shipping: $5.09", "others"])

tag_map = {"Free shipping": 1, "+Shipping: $5.09": 0, "others": 2}
tag_encoded = tag_map[tag]
title_vec = vectorizer.transform([title]).toarray()
df_input = pd.DataFrame(title_vec)
df_input["price"] = price
df_input["tagText"] = tag_encoded

if st.button("Predict"):
    pred = model.predict(df_input)[0]
    st.success(f"📦 Predicted units sold: {int(pred)}")
