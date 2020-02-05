from keras.preprocessing import image
# Operaciones con matrices, tensores, etc
import numpy as np

# Para movernos por el directorio de datos
import zipfile
import os

# Librerías que nos ayudaran a mostrar imagenes
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Importamos nuestro nuevo modelo
from keras.applications.vgg19 import #COMPLETAR
from keras.applications. '''A COPLETAR '''import preprocess_input
from keras.models import Model

# Creamos el modelo recien importado
vgg19_model = #COMPLETAR

# Direccion de nuestra carpeta con las imagenes
dir_dataRandom = './dataSets/DataRandom'


#Obtenemos nuestra imagen para probar la red sin modificar que nos viene mpor defecto
img1 = '''A COMPLETAR'''+'/pato.JPG'

def predecirImg(img_path = ''):
  # Usamos matplot para mostrar la imagen por pantalla y comprobar si la solución tiene sentido
  plt.imshow(mpimg.imread(img_path))

  # Hacemos las modificaciones pertinentes a la imagen para que la red pueda clasificarla
  # Cargamos la imagen reescalandola a las dimensiones que la red necesita
  '''A COMPLETAR'''

  # Preparamos la imagen para un array de numpy.
  '''A COMPLETAR'''

  # Añadimos una dimensión más
  '''A COMPLETAR'''

  # Adecuamos el formato al requerido por el modelo
  x = preprocess_input(x)

  # Predecimos la salida usando el modelo
  predicciones = '''COMPLETAR'''.predict(x)

  # Decodificamos nuestra salida para el primero mejor
  ['''A COMPLETAR'''] = decode_predictions(predicciones, top='''A COMPLETAR''')[0]

  # Mostramos la prediccion por pantalla
  print("Es un: <",descripcion, "> con una probabilidad de: <", "{:.3f}".format(probabilidad*100),"%>")

predecirImg(img1)