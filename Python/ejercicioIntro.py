# Importando las librerias de los modelos
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from keras.preprocessing import image

# Operaciones con matrices, tensores, etc
import numpy as np

# Para movernos por el directorio de datos
import zipfile
import os

# Librerías que nos ayudaran a mostrar imagenes
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Creamos nuestro modelo
model = ResNet50(weights='imagenet')

# Adjunto el dataset (diferente metodo al del colab, ya que está en tu equipo)
dir_data = './dataSets/Data'

# r=root, d=directories, f = files
'''
#EXTRA
for r, d, f in os.walk(dir_data):
  print('Directorios: ',d)
  print('Archivos: ',f)
'''

# Obtenemos nuestra imagen para probar la red sin modificar que nos viene por defecto
img = dir_data+'/Valid/cat/cat.53.jpg'

def predecirImagen(img_path = img):
  # Usamos matplot para mostrar la imagen por pantalla y comprobar si la solución tiene sentido
  plt.imshow(mpimg.imread(img_path))

  # Cargamos la imagen reescalandola a las dimensiones que la red necesita
  img = image.load_img(img_path, target_size=(224, 224)) 

  # Preparamos la imagen para un array de numpy.
  x = image.img_to_array(img)

  # Añadimos una dimensión más
  x = np.expand_dims(x, axis=0)

  # Adecuamos el formato al requerido por el modelo
  x = preprocess_input(x)

  # Predecimos la salida usando el modelo
  predicciones = model.predict(x)

  # Decodificamos la predicción de los tres mejores
  [(clase, descripcion, probabilidad),
  (clase1, descripcion1, probabilidad1),
  (clase2, descripcion2, probabilidad2)] = decode_predictions(predicciones, top=3)[0]

  # Mostramos la prediccion por pantalla
  print('SOLUCION:')
  print('==================================================================================================')
  print("Es un: <",descripcion, "> con una probabilidad de: <", "{:.3f}".format(probabilidad*100),"%>")
  print("Es un: <",descripcion1, "> con una probabilidad de: <", "{:.3f}".format(probabilidad1*100),"%>")
  print("Es un: <",descripcion2, "> con una probabilidad de: <", "{:.3f}".format(probabilidad2*100),"%>")
  print('==================================================================================================')


# Predecimos la imagen
predecirImagen(img)















#===============================================================================================================================
#===============================================================================================================================
# Antes que nada, instalad el WGET: $pip install wget
#===============================================================================================================================
#===============================================================================================================================

import wget

img_URL = 'https://titania.marfeel.com/statics/i/ps/www.ecestaticos.com/imagestatic/clipping/677/f1e/677f1ea2b14769690e8a2cff0eb95428/llevo-a-arreglar-un-coche-de-lujo-asi-le-tomaron-el-pelo-los-mecanicos.jpg?mtime=1556020831'

# Cancer cell
#img_URL = 'https://img.huffingtonpost.com/asset/5b9f2bc125000033003727ee.jpeg?ops=scalefit_720_noupscale'

# Colon cancer
#img_URL = 'https://emlab.uconn.edu/wp-content/uploads/sites/370/2014/10/Colon-Cancer-Cells.jpg'

# Descargamos la imagen
img_from_internet = wget.download(img_URL)

# Predecimos la imagen usando el metodo ya declarado pasandole la imagen descargada
predecirImagen(img_from_internet)