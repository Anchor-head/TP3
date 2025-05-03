# INF7370 Apprentissage automatique
# Travail pratique 3
# Abdoulahat Leye LEYA21309606
# Philip Voinea VOIP85020100
# Évaluation du modèle Autoencodeur
# ***************************************************************************

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')  # Pour éviter les erreurs d'affichage sur serveur
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model, Model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
import os

# ==========================================
# ===============GPU SETUP==================
# ==========================================
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ==========================================
# ==================MODÈLE==================
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

labels = np.array([0] * number_images_class_0 +
                  [1] * number_images_class_1)

image_scale = 128
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
    shuffle=False
)
x = next(generator)

# ***********************************************
# 2) Reconstruire les images tests en utilisant l'autoencodeur
# ***********************************************
reconst_images = autoencoder.predict(x)

fig = plt.figure(figsize=(4, 4))
for i in range(2):
    ax = plt.subplot(2, 2, i + 1)
    ax.set_title("Dauphin (originale)" if i == 0 else "Requin (originale)")
    plt.imshow(x[i * 300])
    ax.axis('off')

    ax = plt.subplot(2, 2, i + 3)
    ax.set_title("Dauphin (reconstruite)" if i == 0 else "Requin (reconstruite)")
    plt.imshow(reconst_images[i * 300])
    ax.axis('off')

plt.tight_layout()
plt.show()
plt.savefig("images_reconstruites.png")
print(" Image des reconstructions sauvegardée sous 'images_reconstruites.png'")

# ***********************************************
# 3) Définir un modèle "encoder" qui est formé de la partie encodeur
# ***********************************************
autoencoder.summary()
embedding_layer=12
encoder_input = autoencoder.input
encoder_output = autoencoder.layers[embedding_layer].output
encoder = Model(encoder_input, encoder_output)
embedding = encoder.predict(x)
embedding_fl = embedding.reshape((number_images, -1))  # Flatten

# ***********************************************
# 4) Normaliser le flattened embedding
# ***********************************************
scaler = StandardScaler()
embedding_nor = scaler.fit_transform(embedding_fl)

# 5) Appliquer un SVM Linéaire sur les images originales (avant l'encodage par le modèle)
# Entrainer le modèle avec le cross-validation
# Afficher la métrique suivante :
#    - Accuracy
x_flat = x.reshape((number_images, -1))  # Applatir les images

svm_img_cv = SVC(kernel='linear', random_state=0, C=1)

print(" Validation croisée en cours sur les images originales...")
scores_img = cross_val_score(svm_img_cv, x_flat, labels, cv=5, scoring='accuracy')

print(" Accuracy moyenne (cross-validation sur images originales) =", round(np.mean(scores_img), 4))
print(" Scores pour chaque fold =", scores_img)

# ***********************************************
# 6) Appliquer un SVC Linéaire avec validation croisée sur le embedding normalisé
# ***********************************************
svm_cv = SVC(kernel='linear', random_state=0, C=1)
print(" Validation croisée en cours sur l'embedding...")
scores = cross_val_score(svm_cv, embedding_nor, labels, cv=5, scoring='accuracy')
print(" Accuracy moyenne (cross-validation 5-fold) =", round(np.mean(scores), 4))
print(" Scores pour chaque fold =", scores)

# ***********************************************
# 7) Appliquer TSNE sur le flattened embedding et l'afficher en 2D
# ***********************************************
print(" Application de t-SNE... (cela peut prendre du temps)")
tsne = TSNE(n_components=2, init='pca', random_state=0)
embedding_tsne = tsne.fit_transform(embedding_fl)

plt.figure(figsize=(6, 6))
scatter=plt.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], c=labels)
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=["Dauphin", "Requin"],
)
plt.title("t-SNE projection des embeddings")
plt.tight_layout()
plt.show()
plt.savefig("tsne_projection.png")
print(" t-SNE sauvegardé sous 'tsne_projection.png'")