# 📉 Telco Customer Churn Prediction

A machine learning project that predicts whether a telecom customer will **leave (churn)** or **stay**, using customer account, service, and billing data.

---

## 📌 Problem Statement

Acquiring a new customer costs far more than keeping an existing one. If a telecom company can identify customers who are likely to leave, it can act early (offers, better plans, support) and reduce revenue loss.

**Goal:** Build a classification model that predicts `Churn` (Yes / No).

---

## 📂 Dataset

- **File:** `Telco-Customer-Churn.csv`
- **Rows:** 7,043 customers
- **Columns:** 21
- **Target:** `Churn` (No = 5,174, Yes = 1,869)

| Group | Columns |
|-------|---------|
| Customer info | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account info | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Charges | `MonthlyCharges`, `TotalCharges` |
| Target | `Churn` |

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-learn
- Jupyter Notebook

---

## 🔄 Project Workflow

### 1. Data Cleaning
- Dropped `customerID` (not useful for prediction).
- Converted `TotalCharges` from text to numeric (`errors='coerce'`).
- Filled the missing `TotalCharges` values with the column mean.

### 2. Exploratory Data Analysis (EDA)
- Gender and churn distribution (donut charts).
- Churn vs Senior Citizen.
- Churn vs Contract type.
- Payment method distribution.
- Monthly charges and tenure distribution for churned vs non-churned customers (KDE plots).

**Key observations from the notebook**
- Customers paying a monthly bill of roughly **80–100** show higher churn.
- Churn is spread almost equally between male and female customers.
- Contract type and tenure show a clear relationship with churn.

### 3. Preprocessing
- Encoded categorical columns to numbers using `LabelEncoder`.
- Checked correlation of every feature with `Churn`.
- Standardized numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) with `StandardScaler` for visual comparison.
- Split data into **75% train / 25% test**.

### 4. Model Building
Six classification models were trained and compared:

| Model | Accuracy (%) |
|-------|--------------|
| **Logistic Regression** | **79.78** |
| Decision Tree (max_depth=5) | 78.88 |
| K-Nearest Neighbors (k=11) | 78.48 |
| Random Forest | 78.25 |
| Naive Bayes | 74.67 |
| SVM | 73.42 |

> Results can change slightly on every run because the train/test split is random (no fixed `random_state`).

### 5. Evaluation
- Accuracy score for every model.
- Confusion matrices for Logistic Regression and Decision Tree.

---

## ✅ Result

**Logistic Regression** gave the best accuracy (~79.8%), closely followed by the Decision Tree with depth 5.

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Nikitesh-Mehare/telco-customer-churn.git
cd telco-customer-churn

# 2. Install the libraries
pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter

# 3. Open the notebook
jupyter notebook Telco_customer_churn.ipynb
```

> Keep `Telco-Customer-Churn.csv` in the same folder as the notebook.

---

## 📁 Project Structure

```
telco-customer-churn/
│── Telco_customer_churn.ipynb   # Full analysis and model code
│── Telco-Customer-Churn.csv     # Dataset
│── README.md                    # Project documentation
```

---

## 🔮 Future Improvements

- Handle class imbalance (SMOTE or class weights).
- Add precision, recall, F1-score and ROC-AUC as main metrics (accuracy alone can mislead on imbalanced data).
- Hyperparameter tuning with GridSearchCV.
- Try XGBoost / Gradient Boosting.
- Deploy as a Flask or Streamlit web app.

---

## 👤 Author

**Nikitesh**
GitHub: [@Nikitesh-Mehare](https://github.com/Nikitesh-Mehare)
