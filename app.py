from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Load the model
model = tf.keras.models.load_model('cifar10_cnn_model.h5')

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    # Read the image
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((32, 32))
    img = np.array(img) / 255.0

    if img.shape != (32, 32, 3):
        return jsonify({'error': 'Invalid image shape'}), 400

    # Predict
    predictions = model.predict(np.expand_dims(img, axis=0))
    predicted_class = class_names[np.argmax(predictions)]
    confidence_scores = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}

    return render_template('result.html', prediction=predicted_class, confidence_scores=confidence_scores)

if __name__ == '__main__':
    app.run(debug=True)
