from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Charger le modèle Keras
model = load_model('mon_modele.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Obtenir le fichier téléchargé
        file = request.files['file']

        # Prétraiter l'image pour la faire correspondre au format d'entrée du modèle
        img = Image.open(file.stream)
        img = img.resize((128, 128))
        img_array = np.array(img)
        img_array = img_array.reshape((1, 128, 128, 3))

        # Faire la prédiction
        prediction = model.predict(img_array)

        # Résultat de la prédiction
        threshold = 0.5  # Choisir un seuil de probabilité pour la classification binaire
        if prediction[0][0] >= threshold:
            result = "Le masque facial est détecté!"
        else:
            result = "Aucun masque facial détecté :)"

        return render_template('index.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
