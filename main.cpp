#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "edgedetector.h"
#include "laplacianZC.h"

// Parametres des trois detecteurs
struct Parametres {
    int detecteur;
    int seuilSobel;
    int seuilLaplace;
    int seuilCannyBas;
    int seuilCannyHaut;
    int indexNoyau;
    int sigmaBlur;
};

// Valeurs de depart des trackbars
Parametres paramsTrackbar = {0, 110, 20, 70, 180, 0, 1};

std::vector<cv::String> fichiersImages;
std::vector<cv::String> fichiersReferences;
int indexImage = 0;
int indexImageChargee = -1;

cv::Mat imageOriginale;
cv::Mat imageReference;

// Titres des fenetres
const std::string nomFenetreReference = "Contours de Reference (dessiner à la main)";
const std::string nomFenetre = "Detecteur de contours | espace/s : lancer l'analyse | q/echap : quitter";

// Mesures demandees pour la comparaison
struct MesuresEvaluation {
    int contours_detectes;
    int contours_reference;
    int contours_corrects;
    int faux_positifs;
    int faux_negatifs;
    double P;
    double TFP;
    double TFN;
};

std::string nomDetecteur(int detecteur) {
    return (detecteur == 0) ? "Sobel" : (detecteur == 1 ? "Laplace" : "Canny");
}

std::string dossierDetecteur(int detecteur) {
    return (detecteur == 0) ? "resultats/sobel/" :
           (detecteur == 1 ? "resultats/laplace/" : "resultats/canny/");
}

std::string descriptionParametres(const Parametres& p) {
    std::string s = nomDetecteur(p.detecteur) + " | ";
    if (p.detecteur == 0)      s += "seuil=" + std::to_string(p.seuilSobel);
    else if (p.detecteur == 1) s += "contraste=" + std::to_string(p.seuilLaplace);
    else                       s += "bas=" + std::to_string(p.seuilCannyBas)
                                  + " haut=" + std::to_string(p.seuilCannyHaut);
    s += " noyau=" + std::to_string(2 * p.indexNoyau + 3)
       + " sigma=" + std::to_string(p.sigmaBlur);
    return s;
}

// Extraction du nom de fichier sans chemin ni extension
std::string extraireNomCourt(const std::string& cheminComplet) {
    size_t posSlash = cheminComplet.find_last_of("/\\");
    std::string nom = (posSlash == std::string::npos) ? cheminComplet : cheminComplet.substr(posSlash + 1);
    size_t posPoint = nom.find_last_of(".");
    return (posPoint == std::string::npos) ? nom : nom.substr(0, posPoint);
}

// Association des images avec leurs references
void apparierImagesReferences(std::vector<cv::String>& fichiersIm,
                              std::vector<cv::String>& fichiersGt) {
    std::vector<cv::String> imOk, gtOk;

    for (size_t i = 0; i < fichiersIm.size(); i++) {
        std::string nomIm = extraireNomCourt(fichiersIm[i]);
        std::string prefixe = nomIm + "_gt";
        bool trouve = false;

        for (size_t j = 0; j < fichiersGt.size(); j++) {
            std::string nomGt = extraireNomCourt(fichiersGt[j]);
            if (nomGt.compare(0, prefixe.size(), prefixe) != 0) continue;

            std::string reste = nomGt.substr(prefixe.size());
            if (reste.empty() || reste[0] == '_') {
                imOk.push_back(fichiersIm[i]);
                gtOk.push_back(fichiersGt[j]);
                trouve = true;
                break;
            }
        }
        if (!trouve) {
            std::cerr << "Pas de reference trouvee pour : " << nomIm << std::endl;
        }
    }

    fichiersIm = imOk;
    fichiersGt = gtOk;
}

// Comptage des pixels de contours
int compterPixelsContours(const cv::Mat& binaire) {
    int compte = 0;
    for (int y = 0; y < binaire.rows; y++) {
        const uchar* rowPtr = binaire.ptr<uchar>(y);
        for (int x = 0; x < binaire.cols; x++) {
            if (rowPtr[x] > 0) {
                compte++;
            }
        }
    }
    return compte;
}

// Detection des contours avec Sobel, Laplace ou Canny
cv::Mat detecterContours(const cv::Mat& img, const Parametres& p) {
    cv::Mat imageTravail = img.clone();
    cv::Mat contours;

    if (p.sigmaBlur > 0) {
        int tailleGauss = 2 * p.sigmaBlur + 1;
        cv::GaussianBlur(img, imageTravail, cv::Size(tailleGauss, tailleGauss), p.sigmaBlur);
    }

    int aperture = 2 * p.indexNoyau + 3;

    if (p.detecteur == 0) {
        EdgeDetector ed;
        ed.setAperture(aperture);
        ed.computeSobel(imageTravail);

        // Adaptation approximative du seuil Sobel au noyau 5x5
        double facteur = 1.0;
        for (int i = 0; i < p.indexNoyau; i++) facteur *= 16.0;
        double seuil = (p.seuilSobel > 0 ? p.seuilSobel : 1) * facteur;

        // Conversion en contours blancs sur fond noir pour les mesures
        cv::Mat mapInv = ed.getBinaryMap(seuil);
        mapInv.convertTo(mapInv, CV_8U);
        contours = 255 - mapInv;

    } else if (p.detecteur == 1) {
        LaplacianZC laplacian;
        laplacian.setAperture(aperture);
        cv::Mat flap = laplacian.computeLaplacian(imageTravail);

        cv::Mat zeros = laplacian.getZeroCrossings(flap);

        // Elimination des passages par zero de faible contraste
        cv::Mat amplitude;
        cv::morphologyEx(imageTravail, amplitude, cv::MORPH_GRADIENT, cv::Mat());
        cv::Mat masqueAmplitude;
        cv::threshold(amplitude, masqueAmplitude, p.seuilLaplace, 255, cv::THRESH_BINARY);

        cv::bitwise_and(zeros, masqueAmplitude, contours);

    } else {
        // Verification des deux seuils de Canny
        int sBas = (p.seuilCannyBas < 1) ? 1 : p.seuilCannyBas;
        int sHaut = (p.seuilCannyHaut <= sBas) ? sBas + 1 : p.seuilCannyHaut;
        cv::Canny(imageTravail, contours, sBas, sHaut, aperture);
    }

    return contours;
}

// Evaluation avec tolerance de 1 pixel et sans double comptage
MesuresEvaluation evaluerContours(const cv::Mat& imageCalculee, const cv::Mat& imageRef) {
    MesuresEvaluation m;

    m.contours_detectes = compterPixelsContours(imageCalculee);
    m.contours_reference = compterPixelsContours(imageRef);
    m.contours_corrects = 0;

    // Deux masques evitent de reutiliser un meme pixel
    cv::Mat detectionUtilisee(imageCalculee.size(), CV_8U, cv::Scalar(0));
    cv::Mat referenceUtilisee(imageRef.size(), CV_8U, cv::Scalar(0));

    // Correspondances entre pixels a la meme position
    for (int y = 0; y < imageCalculee.rows; y++) {
        const uchar* det = imageCalculee.ptr<uchar>(y);
        const uchar* ref = imageRef.ptr<uchar>(y);
        uchar* detUtilisee = detectionUtilisee.ptr<uchar>(y);
        uchar* refUtilisee = referenceUtilisee.ptr<uchar>(y);

        for (int x = 0; x < imageCalculee.cols; x++) {
            if (det[x] > 0 && ref[x] > 0) {
                detUtilisee[x] = 255;
                refUtilisee[x] = 255;
                m.contours_corrects++;
            }
        }
    }

    // Recherche des correspondances restantes dans un voisinage 3x3
    for (int y = 0; y < imageCalculee.rows; y++) {
        const uchar* det = imageCalculee.ptr<uchar>(y);
        uchar* detUtilisee = detectionUtilisee.ptr<uchar>(y);

        for (int x = 0; x < imageCalculee.cols; x++) {
            if (det[x] == 0 || detUtilisee[x] > 0) continue;

            bool trouve = false;
            for (int dy = -1; dy <= 1 && !trouve; dy++) {
                int yy = y + dy;
                if (yy < 0 || yy >= imageRef.rows) continue;

                const uchar* ref = imageRef.ptr<uchar>(yy);
                uchar* refUtilisee = referenceUtilisee.ptr<uchar>(yy);

                for (int dx = -1; dx <= 1; dx++) {
                    int xx = x + dx;
                    if (xx < 0 || xx >= imageRef.cols) continue;

                    if (ref[xx] > 0 && refUtilisee[xx] == 0) {
                        detUtilisee[x] = 255;
                        refUtilisee[xx] = 255;
                        m.contours_corrects++;
                        trouve = true;
                        break;
                    }
                }
            }
        }
    }

    m.faux_positifs = m.contours_detectes - m.contours_corrects;
    m.faux_negatifs = m.contours_reference - m.contours_corrects;

    double denominateur = static_cast<double>(
        m.contours_corrects + m.faux_positifs + m.faux_negatifs);

    if (denominateur > 0) {
        m.P   = m.contours_corrects / denominateur;
        m.TFP = m.faux_positifs / denominateur;
        m.TFN = m.faux_negatifs / denominateur;
    } else {
        m.P = m.TFP = m.TFN = 0.0;
    }

    return m;
}

// Binarisation des references en contours blancs sur fond noir
cv::Mat preparerReference(const cv::Mat& refBrute) {
    cv::Mat refBinaire;
    cv::threshold(refBrute, refBinaire, 127, 255, cv::THRESH_BINARY);

    int nbBlancs = compterPixelsContours(refBinaire);
    if (nbBlancs > (refBinaire.rows * refBinaire.cols) / 2) {
        refBinaire = 255 - refBinaire;
    }
    return refBinaire;
}

// Mise a jour de la detection et des images avec les trackbars
void mettreAJourAffichage(int, void*) {
    if (fichiersImages.empty()) return;

    if (indexImage != indexImageChargee) {
        cv::Mat nouvelleImage = cv::imread(fichiersImages[indexImage], 0);
        cv::Mat nouvelleReference = cv::imread(fichiersReferences[indexImage], 0);

        if (nouvelleImage.empty() || nouvelleReference.empty() ||
            nouvelleImage.size() != nouvelleReference.size()) {
            std::cerr << "Erreur de chargement / taille incompatible pour : "
                      << fichiersImages[indexImage] << std::endl;
            return;
        }

        imageOriginale = nouvelleImage;
        imageReference = preparerReference(nouvelleReference);
        indexImageChargee = indexImage;

        std::string nom = extraireNomCourt(fichiersImages[indexImage]);
        std::cout << "\nIMAGE DE REGLAGE : " << nom << " ("
                  << indexImage + 1 << "/" << fichiersImages.size() << ")" << std::endl;

        // Nom ajoute seulement sur la copie affichee
        cv::Mat affichageOriginal = imageOriginale.clone();
        cv::rectangle(affichageOriginal, cv::Rect(0, 0, affichageOriginal.cols, 31),
                      cv::Scalar(0), -1);
        cv::putText(affichageOriginal, nom + "  (" + std::to_string(indexImage + 1)
                    + "/" + std::to_string(fichiersImages.size()) + ")",
                    cv::Point(8, 22), cv::FONT_HERSHEY_PLAIN, 1.4, cv::Scalar(255), 1);
        cv::imshow("Image Originale", affichageOriginal);
        cv::imshow(nomFenetreReference, 255 - imageReference);
    }

    if (imageOriginale.empty()) return;
    cv::Mat contours = detecterContours(imageOriginale, paramsTrackbar);
    cv::imshow(nomFenetre, 255 - contours);

    MesuresEvaluation m = evaluerContours(contours, imageReference);
    std::cout << "[" << descriptionParametres(paramsTrackbar) << "] -> "
              << "P = " << std::fixed << std::setprecision(3) << m.P
              << " | TFP = " << m.TFP
              << " | TFN = " << m.TFN << std::endl;
}

// Traitement du lot et calcul des moyennes
void traiterLotImages(const std::vector<cv::String>& fichiersIm,
                      const std::vector<cv::String>& fichiersGt,
                      const Parametres& p, bool sauvegarder) {

    const int LARGEUR = 108;
    const int LARGEUR_COMPTAGE = 78;
    std::string dossierSortie = dossierDetecteur(p.detecteur);

    std::cout << "\n" << std::string(LARGEUR, '=') << std::endl;
    std::cout << "TRAITEMENT PAR LOT - " << descriptionParametres(p) << std::endl;
    if (sauvegarder) std::cout << "Sauvegarde dans : " << dossierSortie << std::endl;
    else             std::cout << "Evaluation seule (pas de sauvegarde)" << std::endl;
    std::cout << std::string(LARGEUR, '=') << std::endl;
    std::cout << std::left << std::setw(22) << "Image"
              << std::setw(12) << "Detectes"
              << std::setw(12) << "Reference"
              << std::setw(12) << "Corrects"
              << std::setw(10) << "FP"
              << std::setw(10) << "FN"
              << std::setw(10) << "Perf (P)"
              << std::setw(10) << "TFP"
              << std::setw(10) << "TFN" << std::endl;
    std::cout << std::string(LARGEUR, '-') << std::endl;

    double sommeP = 0.0, sommeTFP = 0.0, sommeTFN = 0.0;
    int nbImagesTraitees = 0;
    size_t nb = std::min(fichiersIm.size(), fichiersGt.size());

    for (size_t i = 0; i < nb; i++) {
        cv::Mat img = cv::imread(fichiersIm[i], 0);
        cv::Mat ref = cv::imread(fichiersGt[i], 0);

        if (!img.data || !ref.data) {
            std::cerr << "Erreur de lecture sur : " << fichiersIm[i] << std::endl;
            continue;
        }

        if (img.size() != ref.size()) {
            std::cerr << "Tailles differentes pour : " << fichiersIm[i] << std::endl;
            continue;
        }

        cv::Mat refPrete = preparerReference(ref);
        cv::Mat contours = detecterContours(img, p);
        MesuresEvaluation m = evaluerContours(contours, refPrete);

        std::string nomCourt = extraireNomCourt(fichiersIm[i]);

        std::cout << std::left << std::setw(22) << nomCourt
                  << std::setw(12) << m.contours_detectes
                  << std::setw(12) << m.contours_reference
                  << std::setw(12) << m.contours_corrects
                  << std::setw(10) << m.faux_positifs
                  << std::setw(10) << m.faux_negatifs
                  << std::fixed << std::setprecision(3)
                  << std::setw(10) << m.P
                  << std::setw(10) << m.TFP
                  << std::setw(10) << m.TFN << std::endl;

        if (sauvegarder) {
            std::string cheminSortie = dossierSortie + nomCourt + "_contours.bmp";
            if (!cv::imwrite(cheminSortie, 255 - contours)) {
                std::cerr << "Echec de sauvegarde : " << cheminSortie
                          << " (le dossier existe-t-il ?)" << std::endl;
            }
        }

        sommeP += m.P;
        sommeTFP += m.TFP;
        sommeTFN += m.TFN;
        nbImagesTraitees++;
    }

    if (nbImagesTraitees > 0) {
        std::cout << std::string(LARGEUR, '-') << std::endl;
        std::cout << std::left
                  << std::setw(LARGEUR_COMPTAGE)
                  << ("MOYENNE SUR " + std::to_string(nbImagesTraitees) + " IMAGES :")
                  << std::fixed << std::setprecision(3)
                  << std::setw(10) << (sommeP / nbImagesTraitees)
                  << std::setw(10) << (sommeTFP / nbImagesTraitees)
                  << std::setw(10) << (sommeTFN / nbImagesTraitees) << std::endl;
    }
    std::cout << std::string(LARGEUR, '=') << "\n" << std::endl;
}

// Programme principal
int main() {
    // Recuperation automatique des images (fonction glob)
    cv::glob("ref/im/*.*", fichiersImages, false);
    cv::glob("ref/gt/*.*", fichiersReferences, false);

    std::sort(fichiersImages.begin(), fichiersImages.end());
    std::sort(fichiersReferences.begin(), fichiersReferences.end());
    apparierImagesReferences(fichiersImages, fichiersReferences);

    if (fichiersImages.empty() || fichiersReferences.empty()) {
        std::cerr << "Erreur : Aucun couple image/reference dans 'ref/im/' et 'ref/gt/' !" << std::endl;
        std::cerr << "Lance le programme depuis le dossier homework3 (make)." << std::endl;
        return -1;
    }

    std::cout << "Nombre de couples image / reference : " << fichiersImages.size() << std::endl;

    // Image de depart choisie pour regler plus facilement les seuils
    for (size_t i = 0; i < fichiersImages.size(); i++) {
        if (extraireNomCourt(fichiersImages[i]) == "golfcart") {
            indexImage = static_cast<int>(i);
            break;
        }
    }

    cv::namedWindow("Image Originale");
    cv::namedWindow(nomFenetreReference);
    cv::namedWindow(nomFenetre);

    // Reglage des detecteurs et choix de l'image avec les trackbars
    cv::createTrackbar("Image (0 a N-1)", nomFenetre, &indexImage,
                       static_cast<int>(fichiersImages.size()) - 1, mettreAJourAffichage);
    cv::createTrackbar("Detecteur 0:Sobel 1:Lapl 2:Canny", nomFenetre, &paramsTrackbar.detecteur, 2, mettreAJourAffichage);
    cv::createTrackbar("SOBEL : seuil", nomFenetre, &paramsTrackbar.seuilSobel, 600, mettreAJourAffichage);
    cv::createTrackbar("LAPLACE : contraste mini", nomFenetre, &paramsTrackbar.seuilLaplace, 255, mettreAJourAffichage);
    cv::createTrackbar("CANNY : seuil bas", nomFenetre, &paramsTrackbar.seuilCannyBas, 500, mettreAJourAffichage);
    cv::createTrackbar("CANNY : seuil haut", nomFenetre, &paramsTrackbar.seuilCannyHaut, 500, mettreAJourAffichage);
    cv::createTrackbar("Noyau 3x3 / 5x5", nomFenetre, &paramsTrackbar.indexNoyau, 1, mettreAJourAffichage);
    cv::createTrackbar("Lissage Gaussien (sigma)", nomFenetre, &paramsTrackbar.sigmaBlur, 5, mettreAJourAffichage);

    mettreAJourAffichage(0, 0);

    std::cout << "\n=== MODE INTERACTIF ===" << std::endl;
    std::cout << "1. Curseur 'Image' : choisis une photo parmi les "
              << fichiersImages.size() << " (son nom est affiche sur l'image)." << std::endl;
    std::cout << "2. Regle Sobel, Laplace et Canny, puis verifie les memes valeurs" << std::endl;
    std::cout << "   sur plusieurs images avec le curseur 'Image'." << std::endl;
    std::cout << "3. Appuie sur ESPACE ou S dans une fenetre OpenCV pour lancer le lot." << std::endl;
    std::cout << "   Appuie sur Q ou ECHAP pour quitter sans traitement." << std::endl;

    // Espace ou S pour lancer le lot, Q ou Echap pour quitter
    while (true) {
        int touche = cv::waitKey(0);
        if (touche == 32 || touche == 's' || touche == 'S') break;
        if (touche == 27 || touche == 'q' || touche == 'Q') {
            cv::destroyAllWindows();
            return 0;
        }
    }
    cv::destroyAllWindows();

    // Application des trois detecteurs a toutes les images
    for (int d = 0; d < 3; d++) {
        Parametres p = paramsTrackbar;
        p.detecteur = d;
        traiterLotImages(fichiersImages, fichiersReferences, p, true);
    }

    std::cout << "Termine ! Verifie resultats/sobel, resultats/laplace et resultats/canny." << std::endl;
    return 0;
}
