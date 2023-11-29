from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        img = Image.open(file)
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = img_array.reshape((1, 224, 224, 3))

        prediction = model.predict(img_array)

        if prediction[0] == 1:
            result = "Le masque facial est détecté!"
        else:
            result = "Aucun masque facial détecté."

        return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
