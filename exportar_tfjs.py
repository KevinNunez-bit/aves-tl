"""
Convierte el modelo entrenado en el clúster a TensorFlow.js para la aplicación web del celular.

Uso:
    python exportar_tfjs.py resultados/modelo_final.h5 resultados/clases.json

Requiere TensorFlow 2.15 y tensorflowjs. El resultado se guarda en docs/modelo/.
"""

import os
import sys
import json
import shutil
import tempfile

import tensorflow as tf
import tensorflowjs as tfjs

ruta_h5 = sys.argv[1] if len(sys.argv) > 1 else "resultados/modelo_final.h5"
ruta_clases = sys.argv[2] if len(sys.argv) > 2 else "resultados/clases.json"
destino = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

modelo = tf.keras.models.load_model(ruta_h5)
# Se agrega softmax al final para que la página reciba probabilidades directamente
salida = tf.keras.layers.Softmax()(modelo.output)
modelo = tf.keras.Model(modelo.input, salida)

with tempfile.TemporaryDirectory() as tmp:
    # Se exporta como "graph model" porque las capas de preprocess_input (TFOpLambda)
    # no son compatibles con tf.loadLayersModel en el navegador
    tf.saved_model.save(modelo, tmp)
    shutil.rmtree(os.path.join(destino, "modelo"), ignore_errors=True)
    tfjs.converters.convert_tf_saved_model(tmp, os.path.join(destino, "modelo"),
                                           quantization_dtype_map={"float16": "*"})  # reduce el tamaño a la mitad

# Los nombres vienen en mayúsculas (ABBOTTS BABBLER); se cambian a formato título
with open(ruta_clases) as f:
    clases = [c.title() for c in json.load(f)]
with open(os.path.join(destino, "clases.json"), "w") as f:
    json.dump(clases, f, ensure_ascii=False)

print("Listo:", len(clases), "clases ->", destino)
for archivo in sorted(os.listdir(os.path.join(destino, "modelo"))):
    print("  ", archivo, os.path.getsize(os.path.join(destino, "modelo", archivo)) // 1024, "KB")
