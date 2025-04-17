# ******************************************************************************
# INF7370 Apprentissage automatique 
# Travail pratique 3
# ===========================================================================

# Dans ce script, on �value l'autoencodeur entrain� dans 1_Modele.py sur les donn�es tests.
# On charge le mod�le en m�moire puis on charge les images tests en m�moire
# 1) On �value la qualit� des images reconstruites par l'autoencodeur
# 2) On �value avec une tache de classification la qualit� de l'embedding
# 3) On visualise l'embedding en 2 dimensions avec un scatter plot

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model, Model
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# ==========================================
# ===============GPU SETUP==================
# ==========================================
'''
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
'''
# ==========================================
# ==================MOD�LE==================
# ==========================================

model_path = "Model.keras"
autoencoder = load_model(model_path)

# ==========================================
# ================VARIABLES=================
# ==========================================

mainDataPath = "donnees/"
datapath = mainDataPath + "test"
number_images = 600
number_images_class_0 = 300
number_images_class_1 = 300
labels = np.array([0] * number_images_class_0 + [1] * number_images_class_1)
image_scale = 64
images_color_mode = "rgb"

# ==========================================
# =========CHARGEMENT DES IMAGES============
# ==========================================

data_generator = ImageDataGenerator(rescale=1. / 255)
generator = data_generator.flow_from_directory(
    datapath,
    color_mode=images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size=number_images,
    class_mode=None,
    shuffle=False)

x = generator.__next__()

# ***********************************************
#                  QUESTIONS
# ***********************************************

reconst_images = autoencoder.predict(x)
reconst_images *= 255

fig = plt.figure(figsize=(4, 4))
for i in range(2):
    ax = plt.subplot(2, 2, i+1)
    if i==0:
       ax.set_title("Dauphin (orignale)")
    else:
       ax.set_title("Requin (orignale)")
    plt.imshow(x[i*300,:,:,:])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 2, i+3)
    if i==0:
       ax.set_title("Dauphin (reconstruite)")
    else:
       ax.set_title("Requin (reconstruite)")
    plt.imshow(reconst_images[i*300])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# ***********************************************
#                  QUESTIONS
# ***********************************************

input_layer_index = 0
output_layer_index = 6
encoder = Model(autoencoder.layers[input_layer_index].input, autoencoder.layers[output_layer_index].output)
embedding = encoder.predict(x)
embedding_fl = embedding.reshape((600,16*16*256))

# ***********************************************
#                  QUESTIONS
# ***********************************************

scaler = StandardScaler()
embedding_nor = scaler.fit_transform(embedding_fl)

# ***********************************************
#                  QUESTIONS
# ***********************************************

svm = SVC(kernel='linear', probability=True, random_state=0, C=1)
x_train, x_test, y_train, y_test = train_test_split(embedding_nor, labels, test_size=0.20)
svm_trained = svm.fit(x_train, y_train)
y_svm = svm.predict(x_test)

print("F1-Macro =", f1_score(y_test, y_svm, average='macro'))
print("F1-Micro =", f1_score(y_test, y_svm, average='micro'))
print("Accuracy =", accuracy_score(y_test, y_svm))

# ***********************************************
#                  QUESTIONS
# ***********************************************

tsne = TSNE(n_components=2, init='pca', random_state=0)
embedding_tsne = tsne.fit_transform(embedding_fl)

plt.figure(figsize=(6, 6))
plt.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], c=labels, cmap='viridis')
plt.colorbar()
plt.title("Visualisation TSNE de l'embedding")
plt.show()