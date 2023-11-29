# Importer les bibliothèques nécessaires
import streamlit as st
from PIL import Image
import numpy as np
import joblib 

model = joblib.load("model.pkl")

# Charger le modèle
#model = load_model('model.h5')

st.title("Détection de Masque Facial")

# Section pour télécharger une image
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
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array.reshape((1, 224, 224, 3))  # Ajouter une dimension pour le lot (batch)

    # Faire la prédiction
    prediction = model.predict(img_array)

    # Afficher le résultat de la prédiction
    if prediction[0] == 1:
        st.success("Le masque facial est détecté!")
    else:
        st.warning("Aucun masque facial détecté.")
