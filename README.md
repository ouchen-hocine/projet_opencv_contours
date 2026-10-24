# Homework 3 - Detection de contours

Projet d'analyse d'image : Sobel, Laplace et Canny.
Veuillez trouvé le compte rendu ainsi que la vidéo dans le dossier compte-rendu/

## Compilation

OpenCV et CMake doivent etre installes. Ouvrir un terminal **dans le dossier `homework3`** :

```bash
make
```

Le `makefile` configure, compile puis lance le programme.

Si  le `makefile` ne marche pas, éxécutez les commandes suivantes :

```bash
mkdir -p build
cd build
cmake ..
make
cd ..
./build/main_prog
```

## Programme

Trois fenetres s'ouvrent : image originale, reference et contours detectes.
Un dernière fenêtre avec les réglages des paramètres (trackbar) afin de traiter les images.

- **Image (0 a N-1)** : change l'image de test et sa reference correspondante sans changer les seuils.

- **Detecteur** : 0 = Sobel, 1 = Laplace, 2 = Canny.

- **SOBEL : seuil** : seuil sur la norme du gradient.

- **LAPLACE : contraste mini** : limite les passages par zero de faible contraste.

- **CANNY : seuil bas / haut** : double seuillage.

- **Noyau 3x3 / 5x5** : taille du noyau des detecteurs.

- **Lissage Gaussien (sigma)** : 0 pour desactiver le lissage.


**Espace** ou **S** dans une fenetre OpenCV : traiter les 40 images avec les reglages choisis pour les 3 detecteurs.

**Q** ou **Echap** : quitter sans traiter le lot.
