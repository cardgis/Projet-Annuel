# digit_recognition/views.py
import base64
import io
import numpy as np
import pymongo
from datetime import datetime
from PIL import Image
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from digit_recognition.ml_model import predict_digit  # Import de la fonction de prédiction

@api_view(['POST'])
def predict_view(request):
    """
    Reçoit: { "image_base64": "...base64..." }
    Retourne: { "prediction": X }
    Stocke l'image + la prédiction dans user_drawings (Mongo).
    """
    try:
        data = request.data
        img_base64 = data.get("image_base64")
        if not img_base64:
            return JsonResponse({"error": "Aucune image fournie"}, status=400)

        # Découper et décoder l'image base64 (si elle comprend un préfixe "data:image/png;base64,")
        if ',' in img_base64:
            img_base64 = img_base64.split(',')[1]
        image_bytes = base64.b64decode(img_base64)

        # Charger l'image, la convertir en niveaux de gris, redimensionner 28x28
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')
        pil_image = pil_image.resize((28, 28))
        img_array = np.array(pil_image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Obtention de la prédiction et de la probabilité depuis le modèle CNN
        predicted_class, probability = predict_digit(img_array)

        # Stocker dans MongoDB
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["mnist_db"]
        user_drawings = db["user_drawings"]
        record = {
            "image_base64": data["image_base64"],
            "predicted_class": predicted_class,
            "probability": probability,
            "timestamp": datetime.utcnow()
        }
        user_drawings.insert_one(record)

        # Renvoyer la réponse au frontend
        return Response({"prediction": predicted_class, "probability": probability})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
