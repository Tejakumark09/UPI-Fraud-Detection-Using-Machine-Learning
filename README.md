🚨 UPI Fraud Detection using Machine Learning
A web application built with Streamlit that detects fraudulent UPI transactions using a pre-trained XGBoost classifier. Users can check individual transactions or upload a CSV for bulk analysis.

🧠 Features

🧾 Single Transaction Check via form input

📄 Bulk Transaction Analysis via CSV upload

📈 Fraud prediction powered by XGBoost ML model

💡 Clean and user-friendly Streamlit interface

📥 Downloadable results for uploaded CSVs



🛠️ Tech Stack

Python

Streamlit

XGBoost

Pandas, NumPy, Altair


📂 Project Structure

bash
Copy
Edit
├── app.py                        # Streamlit app
├── UPI Fraud Detection Final.pkl # Trained ML model
├── sample.csv                    # Sample format for bulk upload
├── requirements.txt              # Python dependencies
└── online-payments-fraud-detection.ipynb # Data analysis / model building notebook


⚙️ How to Run Locally

Clone the repo

bash
Copy
Edit
git clone https://github.com/your-username/upi-fraud-detection.git
cd upi-fraud-detection
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app.py


📊 Sample CSV Format

csv
Copy
Edit
Date,Amount,Transaction_Type,Payment_Gateway,Transaction_State,Merchant_Category
12-05-2024,1250,Purchase,PhonePe,Karnataka,Purchases
...


🧠 Model Info

Trained using XGBoost

Input features include transaction amount, date, type, location, merchant category, and payment gateway

Output: Fraud or Not Fraud



📄 License

This project is open-source and free to use under the MIT License.

