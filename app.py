import os
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from ela_utils import ela_transform, IMG_SIZE

UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model.h5'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # ELA & prediction
            ela_img = ela_transform(image_path).resize(IMG_SIZE)
            ela_array = img_to_array(ela_img) / 255.0
            ela_array = np.expand_dims(ela_array, axis=0)
            pred = model.predict(ela_array)[0][0]

            prediction = "Fake" if pred >= 0.5 else "Real"
            confidence = round(pred * 100 if prediction == "Fake" else (1 - pred) * 100, 2)

    return render_template("index.html", prediction=prediction, confidence=confidence, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
