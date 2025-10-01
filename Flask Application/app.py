# app.py
import os
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Config
UPLOAD_FOLDER = "uploads"
MODEL_WEIGHTS = "vgg_unfrozen.h5"   # the file you tried to load
ALLOWED_EXT = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Build model architecture (must match the weights file if you are loading weights)
def build_model():
    base_model = VGG19(include_top=False, input_shape=(128,128,3))
    x = base_model.output
    flat = Flatten()(x)
    class_1 = Dense(4608, activation='relu')(flat)
    drop_out = Dropout(0.2)(class_1)
    class_2 = Dense(1152, activation='relu')(drop_out)
    output = Dense(2, activation='softmax')(class_2)
    model = Model(base_model.inputs, output)
    return model

# Try to load model in two ways:
# 1) If MODEL_WEIGHTS is a full model (saved with model.save), use load_model
# 2) Otherwise, build architecture and load_weights
model_03 = None
if os.path.isfile(MODEL_WEIGHTS):
    try:
        # Try load as a full model first
        model_03 = load_model(MODEL_WEIGHTS)
        print(f"Loaded full model from '{MODEL_WEIGHTS}'.")
    except Exception as e_full:
        print(f"Not a full-saved model or load_model failed: {e_full}")
        print("Trying to build model architecture and load weights instead...")
        try:
            model_03 = build_model()
            model_03.load_weights(MODEL_WEIGHTS)
            print(f"Weights loaded into architecture from '{MODEL_WEIGHTS}'.")
        except Exception as e_weights:
            print(f"Failed to load weights into the model architecture: {e_weights}")
            model_03 = None
else:
    print(f"Model file '{MODEL_WEIGHTS}' not found. Please place it in the application folder.")
    model_03 = None

if model_03 is None:
    print("Warning: No model is loaded. /predict will return an error until the model file is available.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def get_class_name(class_idx):
    # map index to label; adjust if your model uses different ordering
    return "Normal" if class_idx == 0 else "Pneumonia"

def preprocess_image(path):
    # Read with OpenCV (BGR), convert to RGB, resize, scale
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Unable to read image (cv2.imread returned None). Check the file path or file.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)  # batch dimension
    return image

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model_03 is None:
        return jsonify({"error": "Model not loaded on server. Check server logs and ensure vgg_unfrozen.h5 is present."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(f.filename):
        return jsonify({"error": f"File type not allowed. Allowed: {ALLOWED_EXT}"}), 400

    filename = secure_filename(f.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(file_path)

    try:
        input_img = preprocess_image(file_path)
    except Exception as e:
        return jsonify({"error": f"Image preprocessing failed: {str(e)}"}), 400

    try:
        preds = model_03.predict(input_img)
        class_idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))
        label = get_class_name(class_idx)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

    return jsonify({"label": label, "confidence": confidence})

if __name__ == '__main__':
    # Debug mode for development only
    app.run(host='127.0.0.1', port=5000, debug=True)
