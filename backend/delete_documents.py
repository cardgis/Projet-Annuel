from pymongo import MongoClient

# Se connecter à MongoDB en utilisant l'URL du serveur
client = MongoClient("mongodb://localhost:27017/")

# Accéder à la base de données
db = client['nom_de_votre_base_de_donnees']

# Accéder à la collection
user_drawings = db['user_drawings']

# Supprimer tous les documents
result = user_drawings.delete_many({})
print(f"{result.deleted_count} documents supprimés.")
