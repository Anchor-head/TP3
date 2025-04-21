# INF7370 Apprentissage automatique
# Travail pratique 3

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')  # Important pour les serveurs sans interface graphique
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
from sklearn.decomposition import PCA
import os

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #Philip j'ai desactivé mon GPU ici

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

# ==========================================
# ==========RECONSTRUCTION IMAGES===========
# ==========================================

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
print("L'image des reconstructions a été sauvegardée sous le nom 'images_reconstruites.png'")

# ==========================================
# ==========ENCODEUR========================
# ==========================================

# Affiche la structure du modèle
autoencoder.summary()

# Définir le modèle encodeur correctement
encoder_input = autoencoder.input
encoder_output = autoencoder.get_layer('max_pooling2d_1').output  
encoder = Model(encoder_input, encoder_output)

embedding = encoder.predict(x)
embedding_fl = embedding.reshape((600, 64 * 64 * 256))  # Flatten

# ==========================================
# ======NORMALISATION EMBEDDING=============
# ==========================================

scaler = StandardScaler()
embedding_nor = scaler.fit_transform(embedding_fl)

# ==========================================
# ===============SVM========================
# ==========================================

svm = SVC(kernel='linear', probability=True, random_state=0, C=1)
x_train, x_test, y_train, y_test = train_test_split(embedding_nor, labels, test_size=0.20)
svm_trained = svm.fit(x_train, y_train)
y_svm = svm.predict(x_test)

print("F1-Macro = ", f1_score(y_test, y_svm, average='macro'))
print("F1-Micro = ", f1_score(y_test, y_svm, average='micro'))
print("Accuracy = ", accuracy_score(y_test, y_svm))

# ==========================================
# ================TSNE======================
# ==========================================

# Réduction préalable par PCA à 50 dimensions
print(" Réduction de dimension avec PCA...")
pca = PCA(n_components=50)
embedding_pca = pca.fit_transform(embedding_fl)

# Ensuite t-SNE
print(" Application de t-SNE sur les données réduites...")
tsne = TSNE(n_components=2, init='pca', random_state=0)
embedding_tsne = tsne.fit_transform(embedding_pca)

plt.figure(figsize=(6, 6))
plt.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], c=labels)
plt.colorbar()
plt.title("t-SNE projection des embeddings")
plt.tight_layout()
plt.savefig("tsne_projection.png")
print(" t-SNE sauvegardé sous le nom 'tsne_projection.png'")