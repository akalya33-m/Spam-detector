from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    probability = ""

    if request.method == "POST":
        message = request.form["message"]

        data = vectorizer.transform([message])
        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        prediction = "SPAM ❌" if pred == 1 else "NOT SPAM ✅"
        probability = round(prob * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )

# 🔑 IMPORTANT for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
