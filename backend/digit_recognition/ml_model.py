import tensorflow as tf
import numpy as np
import os

# Charger le modèle CNN depuis le fichier .h5
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modele_cnn.h5")
model = tf.keras.models.load_model(MODEL_PATH)

def predict_digit(img_array):
    """
    Reçoit un array de forme (1, 28, 28) ou (1, 28, 28, 1) correspondant
    à une image en niveaux de gris normalisée (valeurs entre 0 et 1).
    Retourne un tuple (classe_prédite, probabilité_maximale).
    """
    # Si nécessaire, ajuster la forme de l'image. Par exemple :
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=-1)  # Ajoute une dimension pour le canal
    # Assurez-vous que l'input a la bonne forme (1, 28, 28, 1)
    if img_array.shape != (1, 28, 28, 1):
        img_array = img_array.reshape((1, 28, 28, 1))

    probabilities = model.predict(img_array)
    predicted_class = int(np.argmax(probabilities, axis=1)[0])
    probability = float(np.max(probabilities))
    return predicted_class, probability
