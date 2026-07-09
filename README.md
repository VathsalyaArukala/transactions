# Fraud Detection for Online Transaction

A machine learning-based web application that detects fraudulent transactions by analyzing sender details, receiver details, transaction amount, transaction type, and fraud history. The system predicts fraud at three levels: **sender fraud**, **receiver fraud**, and **overall transaction fraud**.

This project is built using **Python, Flask, numpy and Joblib** and provides a simple web interface for entering transaction details and viewing fraud predictions in real time.

---

## Features

- Predicts **sender fraud**
- Predicts **receiver fraud**
- Predicts **overall transaction fraud**
- Displays **fraud probabilities**
- Uses trained machine learning models for prediction
- Simple and user-friendly **Flask web interface**
- Can be extended with confusion matrix and evaluation metrics visualization

---

## Project Overview

Online transactions are increasing rapidly, and so are fraudulent activities. This project helps identify suspicious transactions by training machine learning models on transaction-related features such as sender history, receiver history, transaction amount, and transaction type.

The system takes transaction input from the user and predicts:
- Whether the **sender is fraudulent**
- Whether the **receiver is fraudulent**
- Whether the **overall transaction is fraudulent**

This project is useful for understanding how **machine learning can be applied in fraud detection systems**.

---

## Tech Stack

### Programming Language
- Python

### Framework
- Flask

### Libraries / Tools
- NumPy
- Joblib
- Matplotlib (if used for evaluation graphs)

### Frontend
- HTML
- CSS

---

## Machine Learning Models Used

The project uses separate machine learning models for:
- **Sender Fraud Detection**
- **Receiver Fraud Detection**
- **Overall Fraud Detection**

The trained models are stored as `.pkl` files and loaded inside the Flask application for real-time prediction.

---

## Input Features

The fraud prediction is based on transaction-related features such as:

- `sender_id`
- `receiver_id`
- `amount`
- `transaction_type`
- `sender_fraud_history`
- `receiver_fraud_history`

---

## Output Predictions

The system generates predictions for:

- **Sender Fraud** → Fraud / Not Fraud
- **Receiver Fraud** → Fraud / Not Fraud
- **Overall Fraud** → Fraud / Safe Transaction

---

## Project Structure

```bash
transactions/
│── app.py                        # Flask application
│── transactions.csv              # Dataset used for training/testing
│── generate_transactions.py      # Script for generating or preparing transaction data
│── requirements.txt              # Project dependencies
│── sender_model.pkl              # Trained model for sender fraud prediction
│── receiver_model.pkl            # Trained model for receiver fraud prediction
│── overall_model.pkl             # Trained model for overall fraud prediction
│── encoder.pkl                   # Label encoder for transaction type / categorical data
│
├── templates/
│   └── index.html                # Frontend page for user input and prediction results
│
├── static/
│   ├── style.css                 # Styling for the web page
│   └── images/                   # Optional folder for UI images / graphs
│
└── screenshots/
    ├── home.png              # Home screenshot
    ├── confusion_matrix.png      # Confusion matrix screenshot
    └── model_performance.png               # Accuracy / evaluation metrics screenshot