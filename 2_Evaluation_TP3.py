# ***************************************************************************
# INF7370 Apprentissage automatique 
# Travail pratique 3
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
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os

# ==========================================
# ===============GPU SETUP==================
# ==========================================
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ==========================================
# ==================MODÈLE==================
# ==========================================
model_path = "Model1.keras"
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

image_scale = 256
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
plt.savefig("images_reconstruites.png")
print(" Image des reconstructions sauvegardée sous 'images_reconstruites.png'")

# ***********************************************
# 3) Définir un modèle "encoder" qui est formé de la partie encodeur
# ***********************************************
autoencoder.summary()
encoder_input = autoencoder.input
encoder_output = autoencoder.get_layer('max_pooling2d_1').output
encoder = Model(encoder_input, encoder_output)
embedding = encoder.predict(x)
embedding_fl = embedding.reshape((number_images, -1))  # Flatten

# ***********************************************
# 4) Normaliser le flattened embedding
# ***********************************************
scaler = StandardScaler()
embedding_nor = scaler.fit_transform(embedding_fl)

# ***********************************************
# 5) Appliquer un SVM Linéaire sur les images originales (avant encodage)
# ***********************************************
x_flat = x.reshape((number_images, -1))
svm_img = SVC(kernel='linear', probability=True, random_state=0, C=1)
x_train_img, x_test_img, y_train_img, y_test_img = train_test_split(x_flat, labels, test_size=0.20)
svm_img.fit(x_train_img, y_train_img)
y_pred_img = svm_img.predict(x_test_img)
print(" Accuracy (SVM sur images originales) =", accuracy_score(y_test_img, y_pred_img))

# ***********************************************
# 6) Appliquer un SVC Linéaire sur le flattened embedding normalisé
# ***********************************************
svm_emb = SVC(kernel='linear', probability=True, random_state=0, C=1)
x_train_emb, x_test_emb, y_train_emb, y_test_emb = train_test_split(embedding_nor, labels, test_size=0.20)
svm_emb.fit(x_train_emb, y_train_emb)
y_pred_emb = svm_emb.predict(x_test_emb)
print(" Accuracy (SVM sur embedding) =", accuracy_score(y_test_emb, y_pred_emb))

# ***********************************************
# 7) Appliquer TSNE sur le flattened embedding et l'afficher en 2D
# ***********************************************
print(" Application de t-SNE... (cela peut prendre du temps)")
tsne = TSNE(n_components=2, init='pca', random_state=0)
embedding_tsne = tsne.fit_transform(embedding_fl)

plt.figure(figsize=(6, 6))
plt.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], c=labels)
plt.colorbar()
plt.title("t-SNE projection des embeddings")
plt.tight_layout()
plt.savefig("tsne_projection.png")
print(" t-SNE sauvegardé sous 'tsne_projection.png'")