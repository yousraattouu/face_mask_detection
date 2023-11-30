# Importer les bibliothèques nécessaires
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Charger le modèle Keras
keras_model = load_model('mon_modele.h5')

st.title("Détection de Masque Facial")
st.header("Téléchargez une image")

# Télécharger une image à partir de l'interface utilisateur
uploaded_file = st.file_uploader("Choisissez une image", type="jpg")

# Vérifier si une image a été téléchargée
if uploaded_file is not None:
    # Afficher l'image téléchargée
    st.image(uploaded_file, caption="Image téléchargée.", use_column_width=True)
    st.write("")
    st.write("Classification en cours..")

    # Prétraiter l'image pour la faire correspondre au format d'entrée du modèle
    img = Image.open(uploaded_file)
    img = img.resize((128, 128)) 
    img_array = np.array(img)
    img_array = img_array.reshape((1, 128, 128, 3))
    
    # Faire la prédiction
    prediction = keras_model.predict(img_array)

    # Résultat de la prédiction
    initiale_threshold = 0.5  # Choisir un seuil de probabilité pour la classification binaire

    if prediction[0][0] >= initiale_threshold:
        st.success("Le masque facial est détecté!")
    else:
        st.warning("Aucun masque facial détecté :)")

