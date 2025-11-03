# 🛍️ Mall Customer Segmentation

A data-science project for **customer segmentation** using unsupervised machine learning (K-Means & DBSCAN).  
This project analyzes mall customer data to identify different customer groups based on income and spending patterns.

---

## 📂 Project Structure
customer_segmentation/
│
├─ app.py ← Main Python application script
├─ dbscan_model.joblib ← Pre-trained DBSCAN clustering model
├─ kmeans_model.joblib ← Pre-trained K-Means clustering model
├─ Mall Customer Segment.ipynb ← Jupyter notebook for EDA & modeling
├─ Mall_Customers.csv ← Dataset used for training & analysis
└─ scaler.joblib ← StandardScaler object used for preprocessing


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
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`  
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

## ✅ Key Results & Insights
- Customers were grouped into distinct segments such as:
  - **High Income – High Spending**
  - **Average Income – Average Spending**
  - **Low Income – High Spending (Potential Loyalists)**
  - **Low Income – Low Spending**
- **DBSCAN** identified outliers not belonging to any major group.  
- These results help businesses plan **personalized promotions**, **loyalty programs**, and **targeted campaigns**.

---

## 🚀 How to Use
1. **Clone the repository**
   ```bash
   git clone https://github.com/ahmedessam77/customer-segmentation.git
   cd customer-segmentation
