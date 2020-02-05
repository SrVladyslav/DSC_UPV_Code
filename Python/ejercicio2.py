# Utilidades para manejar las imagenes
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import zipfile
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, MaxPool2D, BatchNormalization

# Modelos para probar
from keras.applications import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.optimizers import Adam

# Adjunto el dataset (diferente metodo al del colab, ya que está en tu equipo)
dir_data = './dataSets/Data'
base_dir = './dataSets/Data'
train_dir = os.path.join(base_dir, 'Train')
valid_dir = os.path.join(base_dir, 'Valid')
test_dir = os.path.join(base_dir, 'Test')

# r=root, d=directories, f = files
'''
#EXTRA
for r, d, f in os.walk(dir_data):
  print('Directorios: ',d)
  print('Archivos: ',f)
'''

# Include_top = False >> se elimina la última capa de salida
preTrainedModelTL = VGG16(weights='imagenet',include_top=False, input_shape = (224,224,3))            #19 Capas
'''
##  Para pruebas si sobra tiempo o en la intimidad del hogar
preTrainedModelTL = ResNet50(weights='imagenet',include_top=False)
preTrainedModelTL = MobileNet(weights='imagenet',include_top=False)
preTrainedModelTL = InceptionResNetV2(weights='imagenet',include_top=False) #164
'''
# Muestro las capas que tiene el modelo
preTrainedModelTL.summary()

# Creamos nuestras modificaciones
t = preTrainedModelTL.output
t = Flatten()(t)
t = Dense(4096, activation='relu')(t)
t = Dropout(0.5)(t)
t = Dense(512, activation = 'relu')(t)
t = Dense(512, activation = keras.layers.LeakyReLU(alpha=0.01))(t)
t = BatchNormalization()(t)

# Última capa con activación Softmax y dos clases de salida (Cat, Dog)
predictions = Dense(2,activation='softmax')(t)

# Creo el modelo
model = Model(inputs = preTrainedModelTL.input, outputs=predictions) 
#model.summary()

print("Capas del modelo: ", len(model.layers))

# Congelamos los pesos de la red para que no se entrene el modelo importado
for layer in model.layers[:18]:  #18 = capas pre-entrenadas
    layer.trainable = False

# ===================================
# Creamos los generadores de imagenes
# ===================================

train_datagen = ImageDataGenerator(preprocessing_function= preprocess_input)

# Data para entrenamiento de la red, lo que la red aprenderá
train_generator = train_datagen.flow_from_directory(train_dir, # PATH al DataSet de entrenamiento
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

# Data de validación, con lo que la red comprobará su perfeccion durante el entrenamiento en cada epoca
# son datos que la red nunca ha visto, entonces por eso la prediccion sera peor
validation_generator = train_datagen.flow_from_directory(valid_dir, 
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

# Data de testeo, la red tampoco la ha visto nunca, la usamos para testear la red al final de todo
test_generator = train_datagen.flow_from_directory(test_dir, 
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

# Compilamos el modelo utilizando el compilador ADAM
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Declaramos los checkpoint para guardar los mejores resultados de los pesos a lo largo del entrenamiento 
from keras.callbacks import ModelCheckpoint

# Guardaremos los mejores pesos aqui
checkpoint_path_loss =  './pesos/cpLoss.ckpt'
checkpoint_path_acc = './pesos/cpAcc.ckpt'

# Pesos del minimo Loss
checkpoint_dirL = os.path.dirname(checkpoint_path_loss)

# Pesos del minimo Accuracy
checkpoint_dirA = os.path.dirname(checkpoint_path_acc)

# Creamos checkpoint  callback 
cp_callback_loss =ModelCheckpoint(checkpoint_path_loss, 
                             monitor = 'val_loss', #val_acc / val_loss
                             save_best_only = True,
                             save_weights_only = True,
                             verbose = 1,
                             period = 1)

cp_callback_acc =ModelCheckpoint(checkpoint_path_acc, 
                             monitor = 'val_acc', #val_acc / val_loss
                             save_best_only = True,
                             save_weights_only = True,
                             verbose = 1,
                             period = 1)

# Entrenamso nuestro modelo
# tamaño de muestras por cada epoca, para que más o menos sea igual
step_size_train = train_generator.n // train_generator.batch_size ## // es una división con redondeo al mínimo
step_size_validation = validation_generator.n // validation_generator.batch_size 


# Entrenamos el modelo
history = model.fit_generator(train_generator,
        steps_per_epoch= step_size_train,
        epochs=15,
        validation_data=validation_generator,
        validation_steps= step_size_validation,
        callbacks=[cp_callback_loss,cp_callback_acc],
        verbose = 1)

# Mostramos las graficas con los resultados
# Añadimos los datos del historico del entrenamiento del modelo para el accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Añadimos los datos del historico del entrenamiento del modelo para el loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Evaluamos nuestro modelo con los datos del Test que la red aún no ha visto
loss, acc = model.evaluate(test_generator)
print('[ORIGINAL]: Loss: ',"{:.4}".format(loss*100),'%, Acc: ',"{:.4}".format(acc*100), '%' )

import cv2
# Vamos a elegir algunas imagenes a boleo
paths =[]

# Cargamos la dirección de nuestras imagenes del test
# dogs
paths.append('./dataSets/Data/Train/dog/dog.4003.jpg') #1
paths.append('./dataSets/Data/Train/dog/dog.4077.jpg') #2
paths.append('./dataSets/Data/Test/dog/dog.125.jpg')   #3
paths.append('./dataSets/Data/Test/dog/dog.604.jpg')   #4
paths.append('./dataSets/Data/Test/dog/dog.133.jpg')   #5
paths.append('./dataSets/Data/Test/dog/dog.226.jpg')   #6
paths.append('./dataSets/Data/Test/dog/dog.232.jpg')   #7
paths.append('./dataSets/Data/Test/dog/dog.303.jpg')   #8
paths.append('./dataSets/Data/Test/dog/dog.286.jpg')   #9
paths.append('./dataSets/Data/Test/dog/dog.374.jpg')   #10
paths.append('./dataSets/Data/Test/dog/dog.357.jpg')   #11
paths.append('./dataSets/Data/Test/dog/dog.180.jpg')   #12
paths.append('./dataSets/Data/Test/dog/dog.383.jpg')   #13
paths.append('./dataSets/Data/Test/dog/dog.473.jpg')   #14
paths.append('./dataSets/Data/Test/dog/dog.476.jpg')   #15
paths.append('./dataSets/Data/Test/dog/dog.480.jpg')   #16
paths.append('./dataSets/Data/Test/dog/dog.562.jpg')   #17
paths.append('./dataSets/Data/Test/dog/dog.593.jpg')   #18

# cats
paths.append('./dataSets/Data/Test/cat/cat.1002.jpg')  #19
paths.append('./dataSets/Data/Test/cat/cat.284.jpg')   #20
paths.append('./dataSets/Data/Test/cat/cat.300.jpg')   #21
paths.append('./dataSets/Data/Test/cat/cat.587.jpg')   #22
paths.append('./dataSets/Data/Test/cat/cat.414.jpg')   #23
paths.append('./dataSets/Data/Test/cat/cat.444.jpg')   #24
paths.append('./dataSets/Data/Test/cat/cat.291.jpg')   #25
paths.append('./dataSets/Data/Test/cat/cat.438.jpg')   #26
paths.append('./dataSets/Data/Test/cat/cat.437.jpg')   #27
paths.append('./dataSets/Data/Test/cat/cat.974.jpg')   #28
paths.append('./dataSets/Data/Test/cat/cat.355.jpg')   #29
paths.append('./dataSets/Data/Test/cat/cat.686.jpg')   #30
paths.append('./dataSets/Data/Test/cat/cat.670.jpg')   #31
paths.append('./dataSets/Data/Test/cat/cat.724.jpg')   #32
paths.append('./dataSets/Data/Test/cat/cat.730.jpg')   #33
paths.append('./dataSets/Data/Test/cat/cat.757.jpg')   #34
paths.append('./dataSets/Data/Test/cat/cat.990.jpg')   #35
paths.append('./dataSets/Data/Test/cat/cat.502.jpg')   #36

# Hacemos la prediccion
# Nuestra funcion de Predicción
def predictByPath(path = './dataSets/Data/Test/dog/dog.980.jpg'):
  # Leemos la imagen
  img = cv2.imread(path)

  # Redimensionamos la imagen a lo que nos pide el modelo
  img1 = cv2.resize(img, (224, 224))

  '''
    La forma de la entrada tiene que ser: [1, image_width, image_height, number_of_channels = 3] 
    3 canales porque tenemos tres colores (RGB)
  '''
  img = np.expand_dims(img1, axis=0)
  #img = np.reshape(img1, [1,224,224,3]) # otra forma de hacerlo

  # Mostramos las imágenes
  plt.imshow(img1)
  plt.show()

  # Obtengo las clases que tiene nuestro generador
  classes = []
  for key, value in train_generator.class_indices.items():
    classes.append(key)
  print("Nuestras clases: ", classes)

  # Hacemos la predicción de la imagen teniendo en cuenta las dos clases que tenemos
  predicciones = model.predict(img, verbose=0)
  claseFinal = predicciones.argmax(axis=-1)

  print('===================================================================')
  print("Cat: ","{:.3f}".format(predicciones[0][0]*100), "% - Dog: ", "{:.3f}".format(predicciones[0][1]*100), "%")
  print("Predicción: ", classes[claseFinal[0]])

# Hagamos las predicciones!
for p in paths:
  predictByPath(p)