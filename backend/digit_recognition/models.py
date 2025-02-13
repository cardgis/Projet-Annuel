import os
import tensorflow as tf
import numpy as np

# Définir BASE_DIR pour pointer vers le dossier racine du projet, par ex.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "modele_cnn.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le fichier modèle n'existe pas à l'emplacement : {MODEL_PATH}")

# Créer une version corrigée d'InputLayer pour gérer 'batch_shape'
class PatchedInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(*args, **kwargs)

# Charger le modèle en utilisant custom_objects pour utiliser PatchedInputLayer et DTypePolicy
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        'InputLayer': PatchedInputLayer,
        'DTypePolicy': tf.keras.mixed_precision.Policy
    },
    compile=False  # Ignorer les paramètres de compilation
)

# Recompiler le modèle avec un optimiseur compatible
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def predict_digit(img_array):
    """
    Prend en entrée un array de forme (1, 784) et retourne la classe prédite (0-9).
    """
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class


# Create your models here.
