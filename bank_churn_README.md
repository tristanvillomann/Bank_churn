# 🏦 Bank Customer Churn Prediction

An end-to-end machine learning pipeline that predicts which bank customers are likely to churn, with an interactive Streamlit dashboard for retention analysts to explore churn drivers and identify high-risk customer segments.



🔗 **[Live Dashboard](https://tristan-villomann-bankchurn.streamlit.app/)** 
---

## 🔍 What I Found

**1. Number of products is the strongest churn predictor — but non-linearly**
Customers with 2 products have the lowest churn rate (8%), while customers with 3-4 products churn at 83-100%. This suggests over-selling products backfires badly — pushing customers past 2 products creates frustration and drives them to competitors. This non-linear relationship is why XGBoost significantly outperformed Logistic Regression.

**2. Inactive members and German customers are highest risk**
IsActiveMember is the second strongest predictor — inactive customers are far more likely to leave. German customers churn at notably higher rates than French or Spanish customers, likely due to stronger local competition in the German market.

**3. Systematic tuning improved recall from 46% to 74%**
The default XGBoost model missed 54% of churning customers. Through two rounds of GridSearchCV and class weight adjustment, recall improved to 74% — meaning the retention team can now proactively reach 3 in 4 customers before they leave.

---

## 📊 Dashboard

The interactive dashboard is built for bank retention analysts and includes:

- **Key metrics** — total customers, churn rate, average age and balance at a glance
- **Churn by variable** — dropdown to explore how any variable affects churn rate
- **High risk segments** — identifies customers combining multiple risk factors with recommended actions

🔗 **[Open Live Dashboard](https://tristan-villomann-bankchurn.streamlit.app/)**

---

## 🗂️ Project Structure

```
bank_churn/
├── data/
│   └── Churn_Modelling.csv        # Download from Kaggle (see below)
├── notebooks/
│   └── churn_analysis.ipynb       # Full analysis notebook
├── figures/
│   ├── confusion_matrix_xgb.png
│   └── feature_importance.png
├── app.py                         # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/bank-churn-prediction.git
cd bank-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
1. Go to: [Kaggle Bank Customer Churn Dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)
2. Download `Churn_Modelling.csv`
3. Place it in the `data/` folder

### 4. Run the dashboard
```bash
streamlit run app.py
```

### 5. Or run the full analysis notebook
```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

---

## 🧠 Methodology

### Dataset
- **10,000 bank customers** across France, Germany and Spain
- **20.37% churn rate** — significantly above the industry average of 10-15%
- 14 features including demographics, account activity and product usage
- Target variable: `Exited` (1 = churned, 0 = stayed)

### Pipeline

```
Raw data
  └─→ Exploratory data analysis
  └─→ Drop identifiers (RowNumber, CustomerId, Surname)
  └─→ Encode categoricals (Geography, Gender) with reference groups
  └─→ Stratified train/test split (80/20)
  └─→ Compare: Logistic Regression, Random Forest, XGBoost
  └─→ Tune XGBoost: GridSearchCV (2 rounds) + scale_pos_weight
  └─→ Evaluate: ROC-AUC, Recall, Precision, Confusion Matrix
  └─→ Deploy: Streamlit dashboard
```

### Why XGBoost?
XGBoost captures non-linear relationships and feature interactions that Logistic Regression cannot. The NumOfProducts finding — where churn spikes at 3-4 products despite a negative linear coefficient — is a clear example of this. Random Forest and XGBoost both handle this, but XGBoost's boosting approach achieved the highest ROC-AUC.

### Handling Class Imbalance
With an 80/20 class split, standard models are biased toward predicting "stayed". We address this with `scale_pos_weight` — set to the ratio of stayed/churned customers (3.93) — which adjusts the loss function to penalise missing churned customers more heavily. This improved recall from 46% to 74% without generating synthetic data.

---

## 📈 Model Comparison

| Model | ROC-AUC | Recall (Churned) |
|---|---|---|
| Logistic Regression | 0.7385 | 0.67 |
| Random Forest | 0.8462 | — |
| XGBoost (default) | 0.8662 | 0.46 |
| **XGBoost (tuned)** | **0.8705** | **0.74** |

---

## 🔧 Final Model Parameters

```python
XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    gamma=0,
    reg_lambda=10,
    scale_pos_weight=3.93,
    random_state=42
)
```

---

## 🚨 Retention Recommendations

| Segment | Churn Rate | Recommended Action |
|---|---|---|
| 🇩🇪 Germany | High | Priority outreach call |
| 😴 Inactive members | High | Re-engagement campaign |
| 📦 3-4 products | 83-100% | Product review meeting |
| 👴 Age 50+ | Above average | Dedicated relationship manager |
| 👩 Female customers | Above average | Targeted retention offer |

**Priority segment:** Inactive customers in Germany with 3+ products — combining three high-risk factors represents the highest churn probability in the model.

---

## 🔧 Tech Stack

- **Python 3.10+**
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — model training, evaluation, GridSearchCV
- `xgboost` — final model
- `statsmodels` — logistic regression summary with p-values
- `imbalanced-learn` — class weight handling
- `matplotlib`, `seaborn` — visualisation
- `streamlit` — interactive dashboard
- `jupyter` — analysis notebook

---

## 🔭 Possible Extensions

- [ ] Add a **customer-level predictor** — input individual customer details and get a churn probability
- [ ] Add **SHAP values** for individual prediction explainability
- [ ] Connect to a **live database** for real-time scoring
- [ ] Add **threshold slider** like the fraud detection dashboard
- [ ] Extend with **survival analysis** to predict *when* a customer will churn, not just if

---

## 💼 Business Relevance

This project mirrors real analytics work at retail banks:

- **Churn prediction** is a core function of any bank's analytics team
- The **retention recommendations** are framed for a non-technical audience
- The **dashboard** gives analysts a self-service tool without needing to run code
- The **model comparison and tuning process** reflects real ML workflow, not just a single model

---

## 👤 Author

**Tristan Villomann**  
Business Analytics, University of Amsterdam  

