#Importamos nuestro nuevo modelo
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

# Creamos el modelo recien importado
vgg19_model = VGG19(weights='imagenet')

# Direccion de nuestra carpeta con las imagenes
dir_dataRandom = './dataSets/DataRandom'

# Obtenemos nuestra imagen para probar la red sin modificar que nos viene mpor defecto
img1 = dir_dataRandom+'/pato.JPG'

def predecirImgSol(img_path = ''):
  # Usamos matplot para mostrar la imagen por pantalla y ver si la solucion tiene algo de sentido
  plt.imshow(mpimg.imread(img_path))

  # Hacemos las modificaciones pertinentes a la imagen para que la red pueda clasificarla
  # Cargamos la imagen reescalandola a las dimensiones que la red necesita
  img = image.load_img(img_path, target_size=(224, 224))

  # Preparamos la imagen para un array de numpy.
  x = image.img_to_array(img)

  # Añadimos una dimensión más
  x = np.expand_dims(x, axis=0)

  # Adecuamos el formato al requerido por el modelo
  x = preprocess_input(x)

  # Predecimos la salida usando el modelo
  predicciones = vgg19_model.predict(x)

  # Decodificamos nuestra salida para el primero mejor
  [(clase, descripcion, probabilidad)] = decode_predictions(predicciones, top=1)[0]

  # Mostramos la prediccion por pantalla
  print("Es un: <",descripcion, "> con una probabilidad de: <", "{:.3f}".format(probabilidad*100),"%>")

predecirImgSol(img1)