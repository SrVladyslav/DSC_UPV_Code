from keras.models import load_model
# Utilidades para manejar las imagenes
import numpy as np
import cv2

# Creamos nuestro modelo que guardamos al entrenar (la estructura)
model = load_model('./pesos/model.h5')

# Inicializamos el modelo con lso pesos aprendidos
model.load_weights('./pesos/model_w.h5')


# Predecimos
# Obtengo las clases que tiene nuestro generador
classes = ['Cat', 'Dog']
print("Nuestras clases: ", classes)
print('=================================================')
# Nuestra funcion de Predicción
def predictByPath(path = '/tmp/Data/Data/Test/dog/dog.980.jpg'):
  # Leemos la imagen
  img = cv2.imread(path)

  # Redimensionamos la imagen a lo que nos pide el modelo
  img1 = cv2.resize(img, (224, 224))
  img = np.expand_dims(img1, axis=0)
  #img = np.reshape(img1, [1,224,224,3]) # otra forma de hacerlo

  # Hacemos la predicción de la imagen teniendo en cuenta las dos clases que tenemos
  predicciones = model.predict(img, verbose=0)
  claseFinal = predicciones.argmax(axis=-1)

  print("Cat: ","{:.3f}".format(predicciones[0][0]*100), "% - Dog: ", "{:.3f}".format(predicciones[0][1]*100), "%")
  print("Prediction: ", classes[claseFinal[0]], '\n')

# Tiene que devolver Gato
predictByPath('./photos/cat.jpg')

# Tiene que devolver Perro
predictByPath('./photos/dog.jpg')

# Tiene que devolver Gato
predictByPath('./photos/tiger.jpg')

# Tiene que devolver o Gato o Perro
predictByPath('./photos/cancer.jpeg')

print('=================================================')


# ==========================================================
# Si te dan problemas de cv2: 'pip install opencv-python'
# ==========================================================