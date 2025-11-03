# 🛍️ Mall Customer Segmentation

A data-science project for **customer segmentation** using unsupervised machine learning (K-Means & DBSCAN).  
This project analyzes mall customer data to identify different customer groups based on income and spending patterns.

---




---

## 🎯 Objective
The main objective of this project is to **segment mall customers** into meaningful groups using clustering techniques.  
These segments can help businesses understand customer behavior and develop targeted marketing strategies.

---

## 📊 Dataset Description
**File:** `Mall_Customers.csv`

| Feature | Description |
|----------|-------------|
| **CustomerID** | Unique ID assigned to each customer |
| **Gender** | Gender of the customer |
| **Age** | Age of the customer |
| **Annual Income (k$)** | Annual income in thousands of dollars |
| **Spending Score (1–100)** | Spending score assigned by the mall based on behavior and spending patterns |

---

## 🧠 Techniques & Tools Used
- **Language:** Python  
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`, `streamlit`
- **Algorithms:**
  - **K-Means Clustering** – for grouping customers based on feature similarity  
  - **DBSCAN** – for density-based clustering and noise detection  
- **Feature Scaling:** Standardization using `StandardScaler` (stored in `scaler.joblib`)

---

## 🧭 Workflow Overview
1. **Data Exploration & Visualization** – Performed in `Mall Customer Segment.ipynb` to analyze distributions and relationships.  
2. **Preprocessing** – Cleaned and standardized numerical data.  
3. **Model Training** – Trained K-Means and DBSCAN models, evaluated with silhouette score and visual plots.  
4. **Model Saving** – Stored trained models and scaler using `joblib`.  
5. **Application Layer** – `app.py` loads models/scaler and predicts cluster labels for new customers.  

---

---

## 🔍 Why K = 5?

| K | Silhouette | Davies-Bouldin | Status |
|---|-------------|----------------|--------|
| 4 | 0.494 | 0.710 | Good |
| **5** | **0.555** | **0.572** | ✅ **Best** |
| 6 | 0.540 | 0.655 | Declining |

✅ **K = 5** was selected because it achieved:
- The **highest Silhouette Score (0.555)** → best cluster separation.  
- The **lowest Davies-Bouldin Index (0.572)** → most distinct clusters.  
- Balanced number of customers per group with clear behavioral differences.

---

## 👥 Customer Segments

| Cluster | Description | Profile | Strategy |
|----------|-------------|----------|-----------|
| **0** 💼 | Low Income, Low Spenders | Budget-conscious customers | Discounts, value offerings |
| **1** 💎 | High Income, Low Spenders | Wealthy but selective | Premium quality products |
| **2** 🎯 | Low Income, High Spenders | Enthusiastic shoppers | Payment plans, trendy items |
| **3** ⭐ | High Income, High Spenders | VIP customers (most valuable) | Luxury items, VIP programs |
| **4** 🌟 | Moderate Spenders | Balanced middle-market | Seasonal promotions |

---

## 📊 Dashboard Pages

| Page | Description |
|-------|-------------|
| **Overview** | Key metrics and overall project summary |
| **Data Exploration** | Interactive visualizations and distributions |
| **Clustering Results** | 5-cluster visualization for K=5 |
| **Customer Insights** | Detailed segment profiles & business strategies |
| **Predict Cluster** | Enter or upload new customer data to predict cluster |

---

## 📈 Key Results

- 🧍 **200 customers** segmented into **5 distinct clusters**  
- 🌀 **Silhouette Score:** 0.555 → strong cluster separation  
- 📉 **Davies-Bouldin Index:** 0.572 → low intra-cluster variance  
- 💰 **Income vs Spending:** weak correlation → suitable for clustering  
- 🌟 **20% of customers** belong to **Cluster 3 (High Value Group)**  
- Streamlit dashboard provides **clear visualization and insights**

---

## 📂 Project Structure

```plaintext
customer_segmentation/
│
├─ app.py                        ← Main Python application script
├─ dbscan_model.joblib           ← Pre-trained DBSCAN clustering model
├─ kmeans_model.joblib           ← Pre-trained K-Means clustering model
├─ Mall Customer Segment.ipynb   ← Jupyter notebook for EDA & modeling
├─ Mall_Customers.csv            ← Dataset used for training & analysis
└─ scaler.joblib                 ← StandardScaler object used for preprocessing

   cd customer-segmentation
