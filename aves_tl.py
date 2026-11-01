"""
Clasificación de 525 especies de aves con transferencia de aprendizaje (MobileNetV2).

Uso:
    python aves_tl.py RUTA_DATASET

RUTA_DATASET debe contener las carpetas train/, valid/ y test/, con una subcarpeta por especie
(dataset "BIRDS 525 SPECIES" de Kaggle).

Procedimiento:
    1. Se carga MobileNetV2 preentrenada en ImageNet, sin su capa de clasificación, y se congela.
    2. Se agrega una capa nueva (GlobalAveragePooling + Dropout + Dense de 525) y se entrena solo esa.
    3. Se descongelan las últimas capas de MobileNetV2 (desde la 130) y se afinan con un learning rate menor.
    4. Se evalúa con el conjunto de prueba y se guardan las gráficas, la matriz de confusión y el modelo.
"""

import os
import sys
import json
import time
import datetime
import zoneinfo

import numpy as np
import matplotlib
matplotlib.use("Agg")  # el clúster no tiene pantalla; las gráficas solo se guardan como imagen
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.utils import image_dataset_from_directory

zona_hermosillo = zoneinfo.ZoneInfo("America/Hermosillo")

def hora():
    return datetime.datetime.now(zona_hermosillo).strftime("%H:%M:%S")

directorio = sys.argv[1] if len(sys.argv) > 1 else "Dataset_Aves"
salida = "resultados"
os.makedirs(salida, exist_ok=True)

print("tensorflow", tf.__version__)
print("dispositivos:", tf.config.list_physical_devices())
print("Empece a las", hora())
t0 = time.time()

BATCH_SIZE = 64
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)

train_dataset = image_dataset_from_directory(os.path.join(directorio, "train"),
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             seed=42)
class_names = train_dataset.class_names

# Se usan los nombres de train para que las clases tengan el mismo índice en los tres conjuntos
validation_dataset = image_dataset_from_directory(os.path.join(directorio, "valid"),
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE,
                                                  class_names=class_names)
test_dataset = image_dataset_from_directory(os.path.join(directorio, "test"),
                                            shuffle=False,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE,
                                            class_names=class_names)

num_clases = len(class_names)
print("Numero de clases:", num_clases)

# Los nombres de las especies se guardan para usarlos en la aplicación web
with open(os.path.join(salida, "clases.json"), "w") as f:
    json.dump(class_names, f, indent=1)

# Aumento de datos ligero (volteo horizontal y rotación pequeña), solo para entrenamiento
aumento = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
])

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.map(lambda x, y: (aumento(x, training=True), y),
                                  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)

# ---------- Modelo: MobileNetV2 congelada + capa de clasificación nueva ----------
pretrained_model_MobileNetV2 = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                                  include_top=False,
                                                                  weights="imagenet")
pretrained_model_MobileNetV2.trainable = False

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = pretrained_model_MobileNetV2(x, training=False)
average_pooling = GlobalAveragePooling2D()(x)
drop = Dropout(0.2)(average_pooling)
# Una salida por especie; sin softmax porque la pérdida usa from_logits=True
outputs = Dense(num_clases, activation="linear", name="prediccion")(drop)

modelo = tf.keras.Model(inputs=inputs, outputs=outputs)
modelo.summary()

modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=["accuracy"])

# Guarda el modelo con mejor val_accuracy durante todo el entrenamiento
checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(salida, "mejor_modelo.h5"),
                                                monitor="val_accuracy",
                                                save_best_only=True,
                                                verbose=1)

# ---------- Etapa 1: entrenamiento de la capa nueva ----------
initial_epochs = 10
print("\nEntrenando la capa nueva, empece a las", hora())
history = modelo.fit(train_dataset,
                     validation_data=validation_dataset,
                     epochs=initial_epochs,
                     callbacks=[checkpoint],
                     verbose=2)

# ---------- Etapa 2: ajuste fino (fine tuning) desde la capa 130 ----------
modelo_base = pretrained_model_MobileNetV2
modelo_base.trainable = True
fine_tune_at = 130
for layer in modelo_base.layers[:fine_tune_at]:
    layer.trainable = False
print("Capas de MobileNetV2:", len(modelo_base.layers), "| afinando desde la", fine_tune_at)

# Es necesario recompilar para que se apliquen los cambios de trainable.
# Se usa un learning rate 10 veces menor para no dañar los pesos preentrenados.
modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=["accuracy"])
modelo.summary()

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs
print("\nAfinando, empece a las", hora())
history_fine = modelo.fit(train_dataset,
                          validation_data=validation_dataset,
                          epochs=total_epochs,
                          initial_epoch=history.epoch[-1] + 1,
                          callbacks=[checkpoint],
                          verbose=2)

modelo.save(os.path.join(salida, "modelo_final.h5"))

# ---------- Gráficas de accuracy y loss ----------

acc = history.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]
loss = history.history["loss"] + history_fine.history["loss"]
val_loss = history.history["val_loss"] + history_fine.history["val_loss"]
epocas = range(1, total_epochs + 1)

with open(os.path.join(salida, "historial.json"), "w") as f:
    json.dump({"accuracy": acc, "val_accuracy": val_acc, "loss": loss, "val_loss": val_loss}, f)

plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(epocas, acc, label="Training Accuracy")
plt.plot(epocas, val_acc, label="Validation Accuracy")
plt.ylim([0, 1.1])
plt.grid(True)
plt.axvline(initial_epochs + 0.5, color="gray", linestyle="--", label="Start Fine Tuning")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy - MobileNetV2 (aves)")

plt.subplot(2, 1, 2)
plt.plot(epocas, loss, label="Training Loss")
plt.plot(epocas, val_loss, label="Validation Loss")
plt.ylim([0, max(max(loss), max(val_loss)) + 0.1])
plt.grid(True)
plt.axvline(initial_epochs + 0.5, color="gray", linestyle="--", label="Start Fine Tuning")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.savefig(os.path.join(salida, "acc_loss.png"), dpi=120, bbox_inches="tight")
plt.close()

# ---------- Evaluación (el conjunto de prueba no se usó durante el entrenamiento) ----------
for nombre, ds in [("Train", train_dataset), ("Validation", validation_dataset), ("Test", test_dataset)]:
    Loss, Accuracy = modelo.evaluate(ds, verbose=0)
    print(f"{nombre:>10}  loss: {Loss:.5f}  accuracy: {Accuracy * 100:.2f}%")

y_real = np.concatenate([y.numpy() for _, y in test_dataset])
y_pred = np.argmax(modelo.predict(test_dataset, verbose=0), axis=1)

reporte = classification_report(y_real, y_pred, labels=range(num_clases),
                                target_names=class_names, zero_division=0)
print(reporte)
with open(os.path.join(salida, "reporte_test.txt"), "w") as f:
    f.write(reporte)

# ---------- Matriz de confusión ----------
# Con 525 clases no se alcanzan a leer los nombres, pero se observa la diagonal
cm = confusion_matrix(y_real, y_pred, labels=range(num_clases))
np.save(os.path.join(salida, "matriz_confusion.npy"), cm)

plt.figure(figsize=(12, 12))
plt.imshow(cm, cmap="Blues", vmax=5)  # hay 5 imágenes por especie en test
plt.colorbar(fraction=0.046)
plt.title(f"Matriz de confusion en test ({num_clases} especies)")
plt.xlabel("Prediccion")
plt.ylabel("Real")
plt.savefig(os.path.join(salida, "matriz_confusion.png"), dpi=150, bbox_inches="tight")
plt.close()

# Matriz reducida con las 20 especies con más errores, para poder leer los nombres
errores = cm.copy()
np.fill_diagonal(errores, 0)
fallas = errores.sum(axis=1)
peores = [i for i in np.argsort(fallas)[::-1][:20] if fallas[i] > 0]
if peores:
    # Columnas: las mismas especies más la especie con la que más se confundió cada una
    columnas = sorted(set(peores) | {int(np.argmax(errores[i])) for i in peores})
    sub = cm[np.ix_(peores, columnas)]
    plt.figure(figsize=(max(8, len(columnas) * 0.5), max(6, len(peores) * 0.45)))
    plt.imshow(sub, cmap="Blues")
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            if sub[i, j] > 0:
                plt.text(j, i, sub[i, j], ha="center", va="center", fontsize=8)
    plt.yticks(range(len(peores)), [class_names[i] for i in peores], fontsize=8)
    plt.xticks(range(len(columnas)), [class_names[i] for i in columnas], rotation=90, fontsize=8)
    plt.title("Especies donde mas se equivoco (test)")
    plt.xlabel("Prediccion")
    plt.ylabel("Real")
    plt.savefig(os.path.join(salida, "matriz_confusion_peores.png"), dpi=150, bbox_inches="tight")
    plt.close()

print("Termine a las", hora(), f"- tardo {(time.time() - t0) / 60:.1f} minutos")
