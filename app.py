from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

model = tf.keras.models.load_model("model.h5")

class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

@app.route("/")
def home():
    return render_template("index.html")   # IMPORTANT

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    if file:
        img = Image.open(file).resize((224,224))
        img = np.array(img)/255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        result = class_names[np.argmax(pred)]

        return f"Prediction: {result}"

    return "No file uploaded"

if __name__ == "__main__":
    app.run(debug=True)