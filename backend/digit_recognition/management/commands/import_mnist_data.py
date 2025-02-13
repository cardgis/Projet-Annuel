import os
import base64
import io
import numpy as np
import pandas as pd
import pymongo
from django.core.management.base import BaseCommand
from PIL import Image
from digit_recognition.ml_model import model  # Assurez-vous du bon chemin

class Command(BaseCommand):
    help = 'Charge les données MNIST dans MongoDB'

    def handle(self, *args, **kwargs):
        # BASE_DIR correspond au dossier "backend"
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        # Chemin vers le dossier script dans digit_recognition
        SCRIPTS_DIR = os.path.join(BASE_DIR, "digit_recognition", "scripts")
        
        # Chemins complets vers les fichiers CSV
        train_csv_path = os.path.join(SCRIPTS_DIR, "mnist_train.csv")
        test_csv_path = os.path.join(SCRIPTS_DIR, "mnist_test.csv")

        # Connexion à MongoDB en local (port par défaut: 27017)
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["mnist_db"]

        # Accéder aux collections
        user_drawings = db["user_drawings"]
        train_col = db["mnist_train"]
        test_col = db["mnist_test"]

        # Chargement des données CSV
        df_train = pd.read_csv(train_csv_path)
        df_test = pd.read_csv(test_csv_path)

        # Insérer les données dans les collections "train" et "test"
        train_col.insert_many(df_train.to_dict("records"))
        test_col.insert_many(df_test.to_dict("records"))

        self.stdout.write(self.style.SUCCESS("Les données MNIST ont été insérées avec succès dans MongoDB"))

        # Exemple pour insérer une image dessinée par un utilisateur et obtenir la prédiction réelle du modèle
        # Ici, 'image_data' contient une chaîne base64 représentant l'image. Assurez-vous que la chaîne est valide.
        example_image = {
            'image_data': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIA...',  # Remplacez par une image base64 réelle
        }

        # Prétraitement de l'image : décodage, conversion en niveaux de gris, redimensionnement en 28x28 et normalisation
        img_base64 = example_image['image_data']
        if ',' in img_base64:
            img_base64 = img_base64.split(',')[1]
        image_bytes = base64.b64decode(img_base64)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')
        pil_image = pil_image.resize((28, 28))
        img_array = np.array(pil_image) / 255.0
        # Ajuster la forme selon le modèle (exemple: (1, 28, 28, 1))
        img_array = img_array.reshape(1, 28, 28, 1)

        # Faire la prédiction avec le modèle CNN
        probabilities = model.predict(img_array)
        predicted_class = int(np.argmax(probabilities))
        probability = float(np.max(probabilities))

        # Mettre à jour l'exemple avec la véritable prédiction et la probabilité
        example_image['prediction'] = predicted_class
        example_image['probability'] = probability

        # Insérer l'image avec la prédiction dans la collection "user_drawings"
        user_drawings.insert_one(example_image)
        self.stdout.write(self.style.SUCCESS(
            f"Image utilisateur insérée : Prédiction = {predicted_class} (Probabilité = {probability:.2f})"
        ))
# import os
# from django.core.management.base import BaseCommand
# import pandas as pd
# import pymongo

# class Command(BaseCommand):
#     help = 'Charge les données MNIST dans MongoDB'

#     def handle(self, *args, **kwargs):
#         # BASE_DIR correspond au dossier "backend"
#         BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#         # Chemin vers le dossier script dans digit_recognition
#         SCRIPTS_DIR = os.path.join(BASE_DIR, "digit_recognition", "scripts")
        
#         # Chemins complets vers les fichiers CSV
#         train_csv_path = os.path.join(SCRIPTS_DIR, "mnist_train.csv")
#         test_csv_path = os.path.join(SCRIPTS_DIR, "mnist_test.csv")

#         # Connexion à MongoDB en local
#         # MongoDB local (par défaut, il fonctionne sur le port 27017)
#         client = pymongo.MongoClient("mongodb://localhost:27017/")
#         db = client["mnist_db"]

#         # Accéder aux collections
#         user_drawings = db["user_drawings"]
#         train_col = db["mnist_train"]
#         test_col = db["mnist_test"] 

#         # Chargement des données CSV avec les chemins ajustés
#         df_train = pd.read_csv(train_csv_path)
#         df_test = pd.read_csv(test_csv_path)

#         # Insérer les données dans les collections "train" et "test"
#         train_col.insert_many(df_train.to_dict("records"))
#         test_col.insert_many(df_test.to_dict("records"))

#         self.stdout.write(self.style.SUCCESS("Les données MNIST ont été insérées avec succès dans MongoDB"))

#         # Exemple pour insérer des images dessinées par les utilisateurs dans la collection "user_drawings"
#         # Ici, je suppose que vous avez une méthode de prédiction qui génère des résultats pour chaque dessin.
#         # Vous pourriez vouloir utiliser une image dessinée et y faire des prédictions comme suit.

#         # Exemple de dessin d'utilisateur (vous devrez l'ajuster à votre cas spécifique)
#         example_image = {
#             'image_data': 'image_base64_string',  # Le dessin de l'utilisateur sous forme de chaîne base64
#             'prediction': 5  # Résultat de la prédiction (par exemple, la classe prédite)
#         }

#         # Insérer l'image dessinée dans la collection "user_drawings"
#         user_drawings.insert_one(example_image)
#         self.stdout.write(self.style.SUCCESS("Les images dessinées par les utilisateurs ont été insérées avec succès dans MongoDB"))


# import os
# from django.core.management.base import BaseCommand
# import pandas as pd
# import pymongo

# class Command(BaseCommand):
#     help = 'Charge les données MNIST dans MongoDB'

#     def handle(self, *args, **kwargs):
#         # BASE_DIR correspond au dossier "backend"
#         BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#         # Chemin vers le dossier script dans digit_recognition
#         SCRIPTS_DIR = os.path.join(BASE_DIR, "digit_recognition", "scripts")
        
#         # Chemins complets vers les fichiers CSV
#         train_csv_path = os.path.join(SCRIPTS_DIR, "mnist_train.csv")
#         test_csv_path = os.path.join(SCRIPTS_DIR, "mnist_test.csv")

#         # Connexion à MongoDB (ici Atlas)
#         client = pymongo.MongoClient("mongodb+srv://alexandrenasalan1:Giscard.1996@cluster0.z1bhheu.mongodb.net/?retryWrites=true&w=majority")
#         db = client["mnist_db"]

#         # Accéder aux collections
#         user_drawings = db["user_drawings"]
#         train_col = db["mnist_train"]
#         test_col = db["mnist_test"] 

#         # Chargement des données CSV avec les chemins ajustés
#         df_train = pd.read_csv(train_csv_path)
#         df_test = pd.read_csv(test_csv_path)

#         # Insérer les données dans les collections "train" et "test"
#         train_col.insert_many(df_train.to_dict("records"))
#         test_col.insert_many(df_test.to_dict("records"))

#         self.stdout.write(self.style.SUCCESS("Les données MNIST ont été insérées avec succès dans MongoDB"))

#         # Exemple pour insérer des images dessinées par les utilisateurs dans la collection "user_drawings"
#         # Ici, je suppose que vous avez une méthode de prédiction qui génère des résultats pour chaque dessin.
#         # Vous pourriez vouloir utiliser une image dessinée et y faire des prédictions comme suit.

#         # Exemple de dessin d'utilisateur (vous devrez l'ajuster à votre cas spécifique)
#         example_image = {
#             'image_data': 'image_base64_string',  # Le dessin de l'utilisateur sous forme de chaîne base64
#             'prediction': 5  # Résultat de la prédiction (par exemple, la classe prédite)
#         }

#         # Insérer l'image dessinée dans la collection "user_drawings"
#         user_drawings.insert_one(example_image)
#         self.stdout.write(self.style.SUCCESS("Les images dessinées par les utilisateurs ont été insérées avec succès dans MongoDB"))
