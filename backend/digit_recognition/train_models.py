# digit_recognition/train_models.py
import pymongo
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models

def load_data_from_mongo():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    #client = pymongo.MongoClient("mongodb+srv://alexandrenasalan1:Giscard.1996@cluster0.z1bhheu.mongodb.net/")
    db = client["mnist_db"]
    train_data = list(db["mnist_train"].find())
    test_data = list(db["mnist_test"].find())

    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)

    # Séparer X / y
    y_train = df_train["label"]
    X_train = df_train.drop(["_id", "label"], axis=1)
    y_test = df_test["label"]
    X_test = df_test.drop(["_id", "label"], axis=1)

    # Normalisation
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    return X_train, y_train, X_test, y_test

def train_classical_models(X_train, y_train, X_test, y_test):
    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    acc_dt = accuracy_score(y_test, dt_pred)

    # Random Forest (avec GridSearch en exemple)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
    }
    rf = RandomForestClassifier()
    grid_rf = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    rf_pred = best_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, rf_pred)

    # SVM (autre exemple de GridSearch)
    param_grid_svm = {
        'C': [1, 10],
        'kernel': ['linear', 'rbf']
    }
    svm_model = SVC()
    grid_svm = GridSearchCV(svm_model, param_grid_svm, cv=3, n_jobs=-1)
    grid_svm.fit(X_train, y_train)
    best_svm = grid_svm.best_estimator_
    svm_pred = best_svm.predict(X_test)
    acc_svm = accuracy_score(y_test, svm_pred)

    print("Accuracy Decision Tree:", acc_dt)
    print("Accuracy RandomForest:", acc_rf)
    print("Accuracy SVM:", acc_svm)

    # Sélection du meilleur
    best_model = None
    best_acc = max(acc_dt, acc_rf, acc_svm)
    if best_acc == acc_dt:
        best_model = dt
    elif best_acc == acc_rf:
        best_model = best_rf
    else:
        best_model = best_svm

    with open("best_classical_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    return best_model

def train_cnn_model(X_train, y_train, X_test, y_test):
    # Reshape en (N, 28, 28, 1)
    X_train_cnn = np.array(X_train).reshape(-1, 28, 28, 1)
    X_test_cnn = np.array(X_test).reshape(-1, 28, 28, 1)
    y_train_cnn = np.array(y_train)
    y_test_cnn = np.array(y_test)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train_cnn, y_train_cnn, epochs=3, validation_split=0.1)
    test_loss, test_acc = model.evaluate(X_test_cnn, y_test_cnn)
    print("CNN test accuracy:", test_acc)

    # Sauvegarder le modèle Keras
    model.save("best_cnn_model.h5")
    return model

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data_from_mongo()

    print("=== Entraînement modèles classiques ===")
    best_classical = train_classical_models(X_train, y_train, X_test, y_test)

    print("\n=== Entraînement CNN ===")
    best_cnn = train_cnn_model(X_train, y_train, X_test, y_test)

    # Vous pouvez comparer best_classical et best_cnn, puis choisir le final
    # Si vous avez déjà 'mon_modele.h5', vous pouvez l’utiliser directement.
