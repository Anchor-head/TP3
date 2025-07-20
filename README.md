# Entrainement d'un autoencodeur pour une tâche de classification d'images d'animaux

Les fichiers finaux se trouvent dans le dossier PhilipVoinea_AbdoulahatLeye_TP3. Les fichiers en dehors de ce dossier sont des brouillons.

Le document PDF dans le dossier PhilipVoinea_AbdoulahatLeye_TP3 contient le raport lu projet. Vous pouvez consulter son contenu ci-dessous.

# Autoencodeur + SVM pour classification binaire d’images d’animaux

## 🛠️ Montage de l’architecture et entrainement du modèle

### Ensemble de données

L’ensemble de données d’entrainement fournies constitue 3600 images d’animaux (1800 pour chacune des deux classes d’animaux : dauphin et requin). On a réservé 20% de cet ensemble de côté pour la validation, c’est-à-dire 720 images (360 par classe). L’ensemble de données de test constitue 600 images d’animaux, soit 300 pour chacune des deux classes.

Nous n’avons effectué aucun prétraitement des données.

### Paramètres et hyperparamètres

Nous avons utilisé l’optimisateur Adam avec les valeurs attribuées par défaut dans Keras (α = 0.001, β1 = 0.9, β2 = 0.999, ε = 10^-7). Nous avons utilisé l’erreur quadratique (MSE) comme fonction de perte.

Pour l’entrainement, nous avons utilisé des lots de 32. Nous avons entrainé le modèle pendant 100 époques sans arrêt précoce; le modèle a continué de s’améliorer au cours des 100 époques.

### Architecture

L’encodeur est composé de quatre couches de convolution avec filtre 3x3 et du zero-padding, chacune suivie d’une activation ReLU et d’une couche d’échantillonage avec filtre 2x2. La première couche de convolution a une profondeur de 128 filtres, la deuxième couche de convolution a une profondeur de 256 filtres, la troisième couche de convolution a une profondeur de 512 filtres et la quatrième couche de convolution a une profondeur de 1024 filtres.

Le décodeur débute avec quatre couches de convolution avec filtre 3x3 et du zero-padding, chacune de ces couches suivie d’une activation ReLU et d’une couche de suréchantillonage avec filtre 2x2. La première couche de convolution a une profondeur de 1024 filtres, la deuxième couche de convolution a une profondeur de 512 filtres, la troisième couche de convolution a une profondeur de 256 filtres et la quatrième couche de convolution a une profondeur de 128 filtres. Ensuite, une couche de convolution 3x3 avec zero-padding est ajoutée avec une profondeur de 3 filtres, suivie d’une activation sigmoïde pour la sortie.

Aucun dropout ou technique de régularisation, sauf le goulot d’étranglement, n’a été utilisé pour l’autoencodeur.

### Résultats d’entrainement

L’entrainement de l’autoencodeur a duré 21 minutes et 44 secondes sur un GPU L4 de Google Colab. L’erreur minimale fût de 0.0036 sur les données d’entrainement et de 0.00374 sur les données de validation.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/a145e771-953a-4faf-8b53-e3843a4d209b" />

### Justification du choix de l’architecture

Nous nous sommes d’abord inspirés de l’architecture fournie par l’exemple MNIST.
Tout au long de notre expérimentation, nous avons gardé la fonction d’optimisation Adam, la fonction de perte MSE, la fonction ReLU comme fonction d’activation pour les couches cachées et la fonction sigmoïde comme fonction d’activation pour la couche de sortie.

Nous avons commencé par tester un modèle identique au modèle du MNIST, mais avec un input de taille 64x64x3. Cela n’a pas achevé une bonne accuracy; nous avons réussi à beaucoup augmenter le résultat juste en augmentant le nombre de filtres pour les couches de convolution; avec 128 filtres pour la première couche de convolutions de l’encodeur et la dernière couche de convolutions du décodeur et 256 filtres pour la deuxième couche de convolutions de l’encodeur et la première couche de convolutions du décodeur, notre SVM a atteint une accuracy de 69%.

Ensuite, sans changer l’architecture, nous avons augmenté la taille de l’input à 128x128x3 afin d’augmenter la résolution des entrées; cela a amélioré la performance de l’autoencodeur en soi, mais a empiré la séparabilité des embeddings et donc la performance du SVM linéaire. J’ai constaté que cela pourrait être dû à la taille de l’embedding qui a augmentée avec la taille de l’input. En effet, un plus petit goulot d’étranglement oblige le modèle à apprendre les attributs les plus importants; plus les embeddings sont grands, plus ils se rapprochent, à la limite, aux données originales et moins l’autoencodeur est utile.

Pour revenir à la taille originale du goulot d’étranglement tout en augmentant la résolution des inputs à 128x128, nous avons ajoutés une troisième couche de convolutions 3x3 avec 512 filtres à la fin de l’encodeur et au début du décodeur, suivi d’une couche d’échantillonnage 2x2 dans l’encodeur et d’une couche de suréchantillonnage 2x2 dans le décodeur. Ce modèle a produit des embeddings plus séparables que tous les autres modèles, atteignant une précision de 72.17% avec le SVM linéaire.

En augmentant la dimension des entrées à 256x256 et en préservant la taille du goulot en ajoutant une quatrième couche de convolutions 3x3 avec 1024 filtres suivie d’une couche d’échantillonnage 2x2 à la fin de l’encodeur et la même couche de convolutions suivie d’une couche de suréchantillonnage 2x2 au début du décodeur, nous avons atteint une accuracy en SVM de 72.67%.

Finalement, nous avons expérimenté avec le drop-out dans les couches cachées de l’encodeur comme technique de régularisation, qui n’a que réduit la performance du modèle même avec un taux d’extinction de 0.1. L’augmentation des données aussi a seulement empirée les modèles.

Pourtant, l’étranglement du goulot a aidé la performance : nous avons d’abord testé le concept en ajoutant une couche de convolutions 3x3 avec 512 filtres suivi d’une couche d’échantillonnage 2x2 à l’encodeur (et l’inverse au décodeur) du modèle aux inputs de taille 64x64 pour créer un encodeur à trois couches de convolutions 3x3 et d’échantillonnage 2x2. Voyant que cela a amélioré la performance du SVM, nous avons ajouté une couche de convolutions 3x3 avec 1024 filtres et une couche d’échantillonnage au modèle aux inputs de taille 128x128 pour créer notre modèle final. Nous attribuons l’augmentation associée au goulot d’étranglement plus étroit ainsi qu’à l’addition d’une convolution qui permet l’apprentissage de caractéristiques plus abstraites. Nous n’avions pas pu adapter cette solution pour des inputs de taille 256x256 pour créer un modèle à 5 couches de convolutions et d’échantillonnage à cause de limites computationnelles. De toute manière, l’augmentation de la dimension des inputs de 128x128 à 256x256 n’a pas semblé augmenter la performance du SVM de beaucoup.

Nous avons expérimenté avec des couches denses vers le goulot d’étranglement; pourtant, ces modèles ont mal performé, peut-être parce que nos ressources computationnelles ont beaucoup limité la taille du goulot étant donné la complexité explosive des couches entièrement connectées.

## 🎯 Résultats
Accuracy du SVM sur embeddings: 74.17%

Accuracy du SVM sur les images originales : 59%

### Reconstruction des images

<img width="500" alt="image" src="https://github.com/user-attachments/assets/66f6c041-134b-4b92-b0d1-dca0d6370cca" />

### Visualisation t-SNE des encodages

<img width="500" alt="image" src="https://github.com/user-attachments/assets/5fb8bebb-4926-4016-b4c7-87e3734fb74e" />

## 📋 Conclusion

Les difficultés quant au choix de l’architecture sont discutées dans la sous-section « Justification du choix de l’architecture » de la section "Montage de l'architecture et entrainement du modèle".

Notre limitation de temps et de ressources ne nous a pas permis d'explorer certaines avenues d'amélioration du modèle, dont celles-ci:
- Choisir plus judicieusement (par le biais d'un grid search?) les paramètres de la régularization l1 et l2
- Expérimenter avec différentes architectures quant aux couches de convolution
