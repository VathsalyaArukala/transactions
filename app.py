from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

app = Flask(__name__)

# Create static folder
if not os.path.exists("static"):
    os.makedirs("static")

# Load models safely
def load_model(path):
    try:
        return joblib.load(path)
    except:
        return None

sender_model = load_model("models/sender.pkl")
receiver_model = load_model("models/receiver.pkl")
le = load_model("models/encoder.pkl")

# Global storage for live CM
y_true = []
y_pred = []

# Generate confusion matrix
def generate_cm():
    if len(y_true) < 2:
        return None

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap='Blues')
    plt.title("Live Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i][j], ha='center', va='center')

    filename = f"cm_{int(time.time())}.png"
    path = os.path.join("static", filename)

    plt.savefig(path)
    plt.close()

    return filename

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            sender = request.form["sender_id"]
            receiver = request.form["receiver_id"]
            amount = float(request.form["amount"])

            txn_input = request.form["transaction_type"]
            txn = le.transform([txn_input])[0] if le else 0

            sh = int(request.form["sender_fraud_history"])
            rh = int(request.form["receiver_fraud_history"])
            actual=int(request.form["actual_label"])
            # Encode IDs
            s_enc = hash(sender) % 10**6
            r_enc = hash(receiver) % 10**6

            features = np.array([[s_enc, r_enc, amount, txn, sh, rh]])

            # Model predictions (safe fallback if model missing)
            if sender_model:
                sp = sender_model.predict_proba(features)[0][1]
            else:
                sp = np.random.rand()

            if receiver_model:
                rp = receiver_model.predict_proba(features)[0][1]
            else:
                rp = np.random.rand()

            sender_result = "fraud" if sp > 0.5 else "safe"
            receiver_result = "fraud" if rp > 0.5 else "safe"

            overall_result = "fraud" if (sender_result == "fraud" or receiver_result == "fraud") else "safe"

            sender_prob = f"{sp*100:.2f}%"
            receiver_prob = f"{rp*100:.2f}%"
            overall_prob = f"{max(sp, rp)*100:.2f}%"

            # Store values for CM
            pred_label = 1 if overall_result == "fraud" else 0
            y_true.append(actual)
            y_pred.append(pred_label)

            cm_image = generate_cm()
            accuracy = precision = recall = f1  = None

            if len(y_true) >= 2:
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)


                accuracy=round(accuracy*100,2)
                precision=round(precision*100,2)
                recall=round(recall*100,2)
                f1=round(f1*100,2)
            print("Actual: ",actual, "Predicted: ",pred_label)
            print("y_true:",y_true,"y_pred: ",y_pred)
            return render_template("index.html",
                sender_result=sender_result,
                receiver_result=receiver_result,
                overall_result=overall_result,
                sender_prob=sender_prob,
                receiver_prob=receiver_prob,
                overall_prob=overall_prob,
                cm_image=cm_image,
                accuracy = accuracy,
                precision = precision,
                recall = recall,
                f1 = f1
                )

        except Exception as e:
            return f"<h2 style='color:red;'>Error: {str(e)}</h2>"
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)