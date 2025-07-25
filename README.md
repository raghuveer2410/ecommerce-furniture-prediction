
# 🛋️ E-commerce Furniture Sales Prediction

This project uses a dataset of 2,000 furniture product listings from AliExpress to build machine learning models that predict the number of items sold based on product features.

## 📊 Dataset Overview

- **Source**: Scraped from AliExpress using Apify
- **Size**: 2,000 rows
- **Features**:
  - `productTitle`: Name of the product
  - `originalPrice`: Price before discounts (often missing)
  - `price`: Selling price
  - `sold`: Units sold (target variable)
  - `tagText`: Shipping or promotional tags

## 🎯 Objective

Predict the number of units sold based on product details using regression models.

## 🛠️ Tech Stack

- Python (pandas, scikit-learn, seaborn, matplotlib)
- Jupyter Notebook
- Streamlit (for interactive UI)
- Joblib (for saving models)

## 📁 Project Structure

```
ecommerce-furniture-prediction/
├── data/
│   └── ecommerce_furniture_dataset.csv
├── notebook/
│   └── furniture_sales_prediction.ipynb
├── app/
│   └── app.py
├── models/
│   └── rf_model.pkl
├── images/
│   └── plots.png
├── requirements.txt
└── README.md
```

## 📈 EDA Highlights

- Distribution plots for `price` and `sold`
- Tag-based sales comparison
- TF-IDF on product titles for feature encoding

## 🧪 Models Used

- **Linear Regression**
- **Random Forest Regressor**

Metrics evaluated:
- Mean Squared Error (MSE)
- R² Score

## 💻 How to Run the Notebook

1. Clone the repository
2. Add the dataset under `/data/`
3. Run the notebook from `/notebook/`

```bash
jupyter notebook furniture_sales_prediction.ipynb
```

## 🌐 Run the Streamlit App

```bash
cd app/
streamlit run app.py
```

## 🧠 Sample Output

- Linear Regression MSE: 134.5, R²: 0.42
- Random Forest MSE: 102.8, R²: 0.57

## 📦 Model Saving

Random Forest model is saved as `rf_model.pkl` using `joblib`.

```python
import joblib
joblib.dump(model, "models/rf_model.pkl")
```

## 🙌 Acknowledgments

- AliExpress and Apify for dataset source
- Spacejoy (Unsplash) for sample images

---

*This is a beginner-level ML project designed to guide through data preprocessing, feature engineering, modeling, and UI development.*
