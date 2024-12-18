from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import tensorflow as tf
import numpy as np
import cv2
import os
import uuid
import imghdr

app = Flask(__name__)

CORS(app)

# Load models
with open('./models/brain_tumor_model.pkl', 'rb') as file:
    brain_model = pickle.load(file)

oral_model = tf.keras.models.load_model('./models/oral_cancer_mode.keras')

# Helper function to validate image file
def is_valid_image(file_path):
    valid_types = ['jpeg', 'png']
    file_type = imghdr.what(file_path)
    return file_type in valid_types

# Helper function to preprocess brain tumor images
def preprocess_image_brain(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    return img

# Helper function to preprocess oral cancer images
def preprocess_image_oral(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (260, 260))
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    return img

# Brain tumor prediction route
@app.route('/predict/brain', methods=['POST'])
def predict_brain():
    print("Brain tumor prediction endpoint hit!") 
    file = request.files['file']
    if file.content_type not in ['image/jpeg', 'image/png']:
        return jsonify({'error': 'Only JPEG and PNG images are allowed'}), 400

    temp_file = f"temp_{uuid.uuid4().hex}.jpg"
    file.save(temp_file)

    if not is_valid_image(temp_file):
        os.remove(temp_file)
        return jsonify({'error': 'Invalid image file'}), 400

    try:
        img = preprocess_image_brain(temp_file)
        prediction = brain_model.predict(img)
        indices = prediction.argmax()

        result = "No Tumor Detected" if indices == 2 else "Tumor Detected"
        print(result)
        return jsonify({'prediction': result})
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

# Oral cancer prediction route
@app.route('/predict/oral', methods=['POST'])
def predict_oral():
    print("Oral Cancer prediction endpoint hit!") 
    file = request.files['file']
    if file.content_type != 'image/jpeg':
        return jsonify({'error': 'Only JPEG images are allowed for oral cancer'}), 400

    temp_file = f"temp_{uuid.uuid4().hex}.jpg"
    file.save(temp_file)

    if not is_valid_image(temp_file):
        os.remove(temp_file)
        return jsonify({'error': 'Invalid image file'}), 400

    try:
        img = preprocess_image_oral(temp_file)
        prediction = oral_model.predict(img)[0][0]
        result = "No Tumor Detected" if prediction > 0.5 else "Tumor Detected"
        print(result)
        return jsonify({'prediction': result})
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == '__main__':
    app.run(debug=True)
