import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import datetime
from datetime import datetime as dt
import time
import base64
import pickle
from xgboost import XGBClassifier

# Load model
pickle_file_path = "UPI Fraud Detection Final.pkl"
loaded_model = pickle.load(open(pickle_file_path, 'rb'))

# Minimalistic Page Config
st.set_page_config(page_title="UPI Fraud Detector", layout="centered")

# Custom Style
st.markdown("""
    <style>
        html, body {
            background-color: #f8f9fa;
        }
        .main {
            padding: 2rem;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #333;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            margin-top: 1rem;
        }
        .stButton>button:hover {
            background-color: #555;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("UPI Transaction Fraud Detector")
st.markdown("Check if your transaction is fraudulent using our ML model. Input manually or upload a CSV.")

# Dropdown data
tt = ["Bill Payment", "Investment", "Other", "Purchase", "Refund", "Subscription"]
pg = ["Google Pay", "HDFC", "ICICI UPI", "IDFC UPI", "Other", "Paytm", "PhonePe", "Razor Pay"]
ts = ['Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
      'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
      'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan',
      'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
mc = ['Donations and Devotion', 'Financial services and Taxes', 'Home delivery', 'Investment',
      'More Services', 'Other', 'Purchases', 'Travel bookings', 'Utilities']

# Single Transaction Input
st.subheader("🔍 Inspect a Single Transaction")
with st.form("single_transaction_form"):
    tran_date = st.date_input("Transaction Date", datetime.date.today())
    month = tran_date.month
    year = tran_date.year

    col1, col2 = st.columns(2)
    with col1:
        tran_type = st.selectbox("Transaction Type", tt)
        tran_state = st.selectbox("Transaction State", ts)
    with col2:
        pmt_gateway = st.selectbox("Payment Gateway", pg)
        merch_cat = st.selectbox("Merchant Category", mc)

    amt = st.number_input("Transaction Amount", min_value=0.0, step=0.1)

    single_check = st.form_submit_button("Check Single Transaction")

if single_check:
    with st.spinner("Checking transaction..."):
        tt_oh = [1 if x == tran_type else 0 for x in tt]
        pg_oh = [1 if x == pmt_gateway else 0 for x in pg]
        ts_oh = [1 if x == tran_state else 0 for x in ts]
        mc_oh = [1 if x == merch_cat else 0 for x in mc]

        features = [amt, year, month] + tt_oh + pg_oh + ts_oh + mc_oh
        result = loaded_model.predict([features])[0]

        st.success("Transaction Checked!")
        if result == 0:
            st.markdown("✅ This transaction is **not fraudulent**.")
        else:
            st.markdown("⚠️ **Alert!** This transaction is **fraudulent**.")

# OR CSV Upload
st.divider()
st.subheader("📂 Upload CSV for Bulk Check")

sample_df = pd.read_csv("sample.csv")
with st.expander("📎 View Sample CSV Format"):
    st.dataframe(sample_df)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Transactions", df)

    if st.button("Check Uploaded Transactions"):
        with st.spinner("Analyzing..."):
            df[['Month', 'Year']] = df['Date'].str.split('-', expand=True)[[1, 2]]
            df[['Month', 'Year']] = df[['Month', 'Year']].astype(int)
            df.drop(columns=['Date'], inplace=True)

            df = df.reindex(columns=['Amount', 'Year', 'Month', 'Transaction_Type',
                                     'Payment_Gateway', 'Transaction_State', 'Merchant_Category'])

            results = []
            for _, row in df.iterrows():
                tt_oh = [1 if x == row['Transaction_Type'] else 0 for x in tt]
                pg_oh = [1 if x == row['Payment_Gateway'] else 0 for x in pg]
                ts_oh = [1 if x == row['Transaction_State'] else 0 for x in ts]
                mc_oh = [1 if x == row['Merchant_Category'] else 0 for x in mc]

                features = [row['Amount'], row['Year'], row['Month']] + tt_oh + pg_oh + ts_oh + mc_oh
                prediction = loaded_model.predict([features])[0]
                # results.append(prediction)
                results.append("Yes" if prediction == 1 else "No")

            df['fraud'] = results
            st.success("All transactions checked.")
            st.dataframe(df)

            # Download Link
            def download_csv():
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                return f'<a href="data:file/csv;base64,{b64}" download="fraud_results.csv">📥 Download Results</a>'

            st.markdown(download_csv(), unsafe_allow_html=True)
