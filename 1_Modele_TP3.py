# -*- coding: utf-8 -*-
# ******************************************************************************
# INF7370 Apprentissage automatique 
# Travail pratique 3
# ===========================================================================

# #===========================================================================
# Ce mod�le est un Autoencodeur Convolutif entrain� sur l'ensemble de donn�es MNIST afin d'encoder et reconstruire les images des chiffres 2 et 7.
# MNIST est une base de donn�es contenant des chiffres entre 0 et 9 �crits � la main en noire et blanc de taille 28x28 pixels
# Pour des fins d'illustration, nous avons pris seulement deux chiffres 2 et 7
#
# Donn�es:
# ------------------------------------------------
# entrainement : classe '2': 1 000 images | classe '7': images 1 000 images
# validation   : classe '2':   200 images | classe '7': images   200 images
# test         : classe '2':   200 images | classe '7': images   200 images
# ------------------------------------------------

# >>> Ce code fonctionne sur MNIST.
# >>> Vous devez donc intervenir sur ce code afin de l'adapter aux donn�es du TP3.
# >>> � cette fin rep�rer les section QUESTION et ins�rer votre code et modification � ces endroits

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des donn�es dans la m�moire
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Le Model � compiler
from keras.models import Model

# Le type d'optimisateur utilis� dans notre mod�le (RMSprop, adam, sgd, adaboost ...)
# L'optimisateur ajuste les poids de notre mod�le par descente du gradient
# Chaque optimisateur a ses propres param�tres
# Note: Il faut tester plusieurs et ajuster les param�tres afin d'avoir les meilleurs r�sultats

from tensorflow.keras.optimizers import Adam

# Les types des couches utlilis�es dans notre mod�le
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D, Activation, Dropout, Flatten, \
    Dense

# Des outils pour suivre et g�rer l'entrainement de notre mod�le
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

# Configuration du GPU
import tensorflow as tf

# Affichage des graphes
import matplotlib.pyplot as plt

from keras import backend as K
import time

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
#config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
#sess = tf.compat.v1.Session(config=config)
#gpus = tf.config.list_physical_devices('GPU')
#for gpu in gpus:
 #   tf.config.experimental.set_memory_growth(gpu, True)
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
'''


# ==========================================
# ================VARIABLES=================
# ==========================================

# ******************************************************
#                       QUESTION DU TP
# ******************************************************
# 1) Ajuster les variables suivantes selon votre probl�me:
# - mainDataPath
# - training_ds_size
# - validation_ds_size
# - image_scale
# - image_channels
# - images_color_mode
# - fit_batch_size
# - fit_epochs
# ******************************************************

# Le dossier principal qui contient les donn�es
mainDataPath = "donnees/"

# Le dossier contenant les images d'entrainement
trainPath = mainDataPath + "entrainement"

# Le dossier contenant les images de validation
validationPath = mainDataPath + "entrainement"

# Le nom du fichier du mod�le � sauvegarder
model_path = "Model.keras"

# Le nombre d'images d'entrainement
training_ds_size = 2880  # total 2000 (1000 classe: 2 et 1000 classe: 7)
validation_ds_size = 720  # total 400 (200 classe: 2 et 200 classe: 7)


# Configuration des  images
image_scale = 64  # la taille des images
image_channels = 3  # le nombre de canaux de couleurs (1: pour les images noir et blanc; 3 pour les images en couleurs (rouge vert bleu) )
images_color_mode = "rgb"  # grayscale pour les image noir et blanc; rgb pour les images en couleurs
image_shape = (image_scale, image_scale,
               image_channels)  # la forme des images d'entr�es, ce qui correspond � la couche d'entr�e du r�seau

# Configuration des param�tres d'entrainement
fit_batch_size = 32  # le nombre d'images entrain�es ensemble: un batch
fit_epochs = 100  # Le nombre d'�poques

# ==========================================
# ==================MOD�LE==================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS DU TP
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Ajuster les deux fonctions:
# 2) encoder
# 3) decoder
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Couche d'entr�e:
# Cette couche prend comme param�tre la forme des images (image_shape)
input_layer = Input(shape=image_shape)


# Partie d'encodage (qui extrait les features des images et les encode)
def encoder(input):
    x = Conv2D(128, (3, 3), padding='same')(input)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    return encoded


# Partie de d�codage (qui reconstruit les images � partir de leur embedding ou la sortie de l'encodeur)
def decoder(encoded):
    x = Conv2D(256, (3, 3), padding='same')(encoded)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(image_channels, (3, 3), padding='same')(x)
    decoded = Activation('sigmoid')(x)
    return decoded


# D�claration du mod�le:
model = Model(input_layer, decoder(encoder(input_layer)))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# ==========================================
# ==========CHARGEMENT DES IMAGES===========
# ==========================================

training_data_generator = ImageDataGenerator(rescale=1. / 255)
validation_data_generator = ImageDataGenerator(rescale=1. / 255)

training_generator = training_data_generator.flow_from_directory(
    trainPath,
    color_mode =images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size = training_ds_size,
    class_mode ="input")

validation_generator = validation_data_generator.flow_from_directory(
    validationPath,
    color_mode =images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size = validation_ds_size,
    class_mode ="input")

(x_train, _) = training_generator.__next__()
(x_val, _) = validation_generator.__next__()


# ==========================================
# ==============ENTRAINEMENT================
# ==========================================

modelcheckpoint = ModelCheckpoint(filepath=model_path,
                                  monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

start_time = time.time()
autoencoder = model.fit(x_train, x_train,
                       epochs=fit_epochs,
                       batch_size=fit_batch_size,
                       verbose=1,
                       callbacks=[modelcheckpoint],
                       shuffle=False,
                       validation_data=(x_val, x_val))
end_time = time.time()
print("Le temps d'ex�cution est de:", end_time - start_time, "secondes.")

# ==========================================
# ========AFFICHAGE DES RESULTATS===========
# ==========================================

plt.plot(autoencoder.history['loss'])
plt.plot(autoencoder.history['val_loss'])
plt.title('Courbe de perte')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Entrainement', 'Validation'])
plt.grid(True)
plt.show()