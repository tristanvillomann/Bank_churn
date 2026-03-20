import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv("data/Churn_Modelling.csv")
except FileNotFoundError:
    st.error("Dataset not found. Please download from Kaggle and place in data/ folder.")
    st.stop()

st.title("🏦 Bank Churn Analysis Dashboard")
st.markdown("""
This dashboard helps retention analysts at banks to identify customers at risk of churning 
and understand which factors drive churn most strongly.

**Built by:** Tristan Villomann — Business Analytics, University of Amsterdam
""")
st.divider()

df = pd.read_csv('C:/Users/Ideapad5/OneDrive/Documents/Projects/bank_churn/data/Churn_Modelling.csv')

# Key metrics
total = len(df)
churn_rate = df['Exited'].mean()
avg_age = df['Age'].mean()
avg_balance = df['Balance'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{total:,}")
col2.metric("Churn Rate", f"{churn_rate:.1%}")
col3.metric("Avg Age", f"{avg_age:.0f}")
col4.metric("Avg Balance", f"€{avg_balance:,.0f}")


st.divider()
st.subheader("📊 Churn Rate by Variable")

# Let user pick a variable
variable = st.selectbox(
    "Select a variable to analyse",
    ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Tenure']
)

# Calculate churn rate by selected variable
churn_by_var = df.groupby(variable)['Exited'].mean().reset_index()
churn_by_var.columns = [variable, 'Churn Rate']
churn_by_var['Churn Rate'] = churn_by_var['Churn Rate'] * 100

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(churn_by_var[variable].astype(str), churn_by_var['Churn Rate'], color='#E63946')
ax.set_ylabel('Churn Rate (%)')
ax.set_title(f'Churn Rate by {variable}')
ax.axhline(churn_rate * 100, color='black', linestyle='--', label=f'Average ({churn_rate:.1%})')
ax.legend()
plt.tight_layout()
st.pyplot(fig)


st.divider()
st.subheader("🚨 High Risk Customer Segments")

st.markdown("""
Based on the model's feature importance and churn analysis, the following customer 
segments are at highest risk and should be prioritised for retention outreach:
""")

# Define high risk segments
high_risk = df[
    (df['Geography'] == 'Germany') |
    (df['IsActiveMember'] == 0) |
    (df['NumOfProducts'] >= 3) |
    (df['Age'] > 50)
].copy()

high_risk_churn = high_risk['Exited'].mean()

col1, col2 = st.columns(2)
col1.metric("High Risk Customers", f"{len(high_risk):,}")
col2.metric("Churn Rate in Segment", f"{high_risk_churn:.1%}")

st.markdown("""
### ⚠️ Action Required

| Segment | Churn Rate | Recommended Action |
|---|---|---|
| 🇩🇪 Germany | High | Priority outreach call |
| 😴 Inactive members | High | Re-engagement campaign |
| 📦 3-4 products | 83-100% | Product review meeting |
| 👴 Age 50+ | Above average | Dedicated relationship manager |
""")

st.info("""
**Retention team recommendation:** Focus this week's outreach on inactive customers 
in Germany with 3+ products. This segment combines three high-risk factors and 
represents the highest churn probability in the model.
""")