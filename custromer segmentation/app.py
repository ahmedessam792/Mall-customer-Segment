# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Models & Scaler
# -----------------------------
@st.cache_resource
def load_models():
    kmeans = joblib.load('kmeans_model.joblib')
    dbscan = joblib.load('dbscan_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return kmeans, dbscan, scaler

kmeans, dbscan, scaler = load_models()

# -----------------------------
# 2. Load Original Data (for visualization)
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

# Add K-Means clusters (if not saved in CSV)
if 'KMeans_Cluster' not in df.columns:
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    X_scaled = scaler.transform(X)
    df['KMeans_Cluster'] = kmeans.predict(X_scaled)

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.title("Mall Customer Segmentation")
st.markdown("""
Predict which customer segment a new shopper belongs to using **K-Means** and **DBSCAN**.
""")

st.sidebar.header("Enter Customer Details")

income = st.sidebar.slider(
    "Annual Income (k$)",
    min_value=15, max_value=140, value=60, step=1
)

spending = st.sidebar.slider(
    "Spending Score (1-100)",
    min_value=1, max_value=100, value=50, step=1
)

# Prepare input
new_customer = pd.DataFrame({
    'Annual Income (k$)': [income],
    'Spending Score (1-100)': [spending]
})

# Scale
new_scaled = scaler.transform(new_customer)

# Predict
kmeans_pred = kmeans.predict(new_scaled)[0]
dbscan_pred = dbscan.fit_predict(new_scaled)[0]  # DBSCAN: fit_predict on new point

# -----------------------------
# 4. Show Results
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("K-Means Cluster")
    st.metric("Cluster ID", kmeans_pred)
    interpretations = {
        0: "Moderate Income & Spending (Balanced)",
        1: "High Income & High Spending (Premium Target)",
        2: "Low Income & High Spending (Impulse Buyer)",
        3: "High Income & Low Spending (Frugal Rich)",
        4: "Low Income & Low Spending (Budget Shopper)"
    }
    st.write(interpretations.get(kmeans_pred, "Unknown"))

with col2:
    st.subheader("DBSCAN Cluster")
    if dbscan_pred == -1:
        st.metric("Cluster ID", "Noise (Outlier)")
        st.warning("This customer doesn't fit any dense group.")
    else:
        st.metric("Cluster ID", dbscan_pred)
        db_interpret = {
            0: "Mainstream (Average)",
            1: "Low Income & Low Spending",
            2: "High Income & High Spending",
            3: "High Income & Low Spending"
        }
        st.write(db_interpret.get(dbscan_pred, "Other Group"))

# -----------------------------
# 5. Optional: Plot Customer on Map
# -----------------------------
if st.checkbox("Show customer on cluster map", value=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot original clusters
    sns.scatterplot(
        data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
        hue='KMeans_Cluster', palette='tab10', ax=ax, alpha=0.7, legend='full'
    )
    
    # Plot new customer
    ax.scatter(income, spending, color='red', s=200, marker='X', label='New Customer')
    ax.legend(title='K-Means Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title("Customer Location in Segmentation Map")
    
    st.pyplot(fig)