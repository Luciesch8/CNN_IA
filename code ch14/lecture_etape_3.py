#-----------------------------------------------------------------------------------------
# @Auteur : Aurélien Vannieuwenhuyze
# @Entreprise : Junior Makers Place
# @Livre :
# @Chapitre : 13 - Classification d'images
#
# Modules necessaires :
#   TENSORFLOW 1.13.1
#   KERAS 2.2.4
#   OPENCV 3.4.5.20
#   PYTTSX3 2.7.1
#   SCIKIT-LEARN 0.21.1
#   NUMPY 1.16.3
#
# Pour installer un module :
#   Cliquer sur le menu File > Settings > Project:nom_du_projet > Project interpreter > bouton +
#   Dans la zone de recherche en haut à gauche saisir le nom du module
#   Choisir la version en bas à droite
#   Cliquer sur le bouton install situé en bas à gauche
#-----------------------------------------------------------------------------------------


import cv2
import numpy as np

#Module de paroles
import pyttsx3 as pyttsx

#Module Keras permettant l'utilisation de notre réseau de neurones
from keras.models import load_model

#Module de gestion des processus
import threading


#Par défaut on active la lecture de lettre à voix haute
lectureActivee = True

#Temps d'attente en secondes entre chaque lecture de lettre à voix haute
dureeDesactivationLectureDeLettre = 5

#fonction de reactivation de la lecture de lettre à voix haute
def activationLecture():
    print('Activation de la lecture de lettres')
    global lectureActivee
    lectureActivee=True

#dimensions de la zone d'écriture
zoneEcritureLongueurMin = 540
zoneEcritureLongueurMax = 590
zoneEcritureLargeurMin = 300
zoneEcritureLargeurMax = 340

#Initialisation de la voix
print('Initialisation de la voix')
engine = pyttsx.init()

#Choix de la voix en français
voice = engine.getProperty('voices')[0]
engine.setProperty('voice', voice.id)

#test de la voix
engine.say('Mode lecture de lettres activé')
engine.runAndWait()

print('Initialisation du modèle d''aprentissage')

#Chargement du modèle entrainé
cnn_model = load_model('modele/modele_cas_pratiqueV2.h5')
kernel = np.ones((5, 5), np.uint8)

#Tableau de lettres avec leur numéro
lettres = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
           11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T',
           21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 27: '-'}



#Par défaut on choisi que la lettre Z est détectée
prediction = 26

#Par défaut aucune détection n'est faite.
lettrePredite = False


print('Initialisation de la webcam')
webCam = cv2.VideoCapture(0)
if webCam.isOpened():
    longueurWebcam = webCam.get(3)
    largeurWebcam = webCam.get(4)
    print('Résolution:' + str(longueurWebcam) + " X " + str(largeurWebcam))
else:
    print('ERREUR')

while True:

    #Par défaut aucune détection n'est faite.
    lettrePredite = False

    # Capture de l'image dans la variable Frame
    # La variable lectureOK est égale à True si la fonction read() est opérationnelle
    (lectureOK, frame) = webCam.read()

    (grabbed, frame) = webCam.read()
    tsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contours_canny = cv2.Canny(gris, 30, 200)

    contours = cv2.findContours(contours_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    for contour in contours:
        perimetre = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.012 * perimetre, True)
        x, y, w, h = cv2.boundingRect(approx)

        #On encadre la zone d'écriture en fonction des paramètres de longueur et largeur de l'ardoise
        if len(approx) == 4 and h>zoneEcritureLargeurMin and w>zoneEcritureLongueurMin and h<zoneEcritureLargeurMax and w<zoneEcritureLongueurMax:

            #Encadrement de la zone d'écriture
            area = cv2.contourArea(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3);

            # Capture de l'image à partir de la zone d'écriture avec une marge intérieure (padding) de 10
            # pixels afin d'isoler uniquement la lettre
            lettre = gris[y + 10:y + h - 10, x + 10:x + w - 10]

            # On detecte les contours de la lettre  à l'aide de l'algorithme de Canny
            cannyLettre = cv2.Canny(lettre, 30, 200)
            contoursLettre = cv2.findContours(cannyLettre.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

            # S'il existe une lettre de dessinée
            if len(contoursLettre) > 5:

                # Creation d'un tableau pour le stockage de l'image de la lettre
                captureAlphabetTMP = np.zeros((400, 400), dtype=np.uint8)

                # On detecte le plus grand contour (Reverse = True)
                cnt = sorted(contoursLettre, key=cv2.contourArea, reverse=True)[0]

                # On stocke les coordonnées du rectangle de delimitation de la lettre
                xc, yc, wc, hc = cv2.boundingRect(cnt)


                for contourLettre in contoursLettre:
                    area = cv2.contourArea(contour)
                    if area > 1000:

                        # On dessine les contours la lettre pour une meilleure lecture (Trait de 10 px)
                        cv2.drawContours(captureAlphabetTMP, contourLettre, -1, (255, 255, 255), 10)

                        # On capture la lettre et on stock les valeurs  des pixels de la zone capturée ans un tableau
                        captureLettre = np.zeros((400, 400), dtype=np.uint8)
                        captureLettre = captureAlphabetTMP[yc:yc + hc, xc:xc + wc]


                        #Des ombres peuvent être capturée dans la zone d'écriture provoquant alors des erreurs de
                        #reconnaissance. Si une ombre est détectée, une des dimension du tableau de capture est
                        #égale à zéro car aucun contour de lettre n'est détecté
                        affichageLettreCapturee = True
                        if (captureLettre.shape[0] == 0 or captureLettre.shape[1] == 0):
                            print("ERREUR A CAUSE DES OMBRES ! : ")
                            affichageLettreCapturee = False

                        #Si ce n'est pas une ombre, on affiche la lettre capturée à l'écran
                        if affichageLettreCapturee:
                            cv2.destroyWindow("ContoursLettre");
                            cv2.imshow("ContoursLettre", captureLettre)

                            # Redimensionnement de l'image
                            newImage = cv2.resize(captureLettre, (28, 28))
                            newImage = np.array(newImage)
                            newImage = newImage.astype('float32') / 255
                            newImage.reshape(1, 28, 28, 1)

                            # Realisation de la prediction
                            prediction = cnn_model.predict(newImage.reshape(1, 28, 28,1))[0]
                            prediction = np.argmax(prediction)

                            # On indique qu'une lettre a été détectée
                            lettrePredite = True


                if lettrePredite:

                    #On desactive la lecture de lettre à voix haute
                    print('Desactivation de la lecture de lettre ' + str(dureeDesactivationLectureDeLettre) + " secondes")
                    lectureActivee = False

                    #On affiche le numéro de la lettre prédit
                    #On ajoute +1 car la première lettre de l'alphabet a pour valeur 0 dans notre modèle de prédiction
                    #Alors qu'elle a la valeur 1 dans notre tableau de correspondance
                    print("Detection:" + str(lettrePredite))
                    print("Prediction = " + str(prediction))

                    #Lecture à voix haute de la lettre prédit
                    if (lettrePredite and prediction != 26):
                        engine.say('Je lis la lettre ' + str(lettres[int(prediction) + 1]))
                        engine.runAndWait()
                        lettrePredite = False

                    if (lettrePredite and prediction == 26):
                        engine.say('Je ne comprends pas la lettre écrite ')
                        engine.runAndWait()
                        lettrePredite = False

                    #Pause du processus de lecture de la lettre puis appel à la fonction pour la reactivation de la
                    #lecture
                    timer = threading.Timer(dureeDesactivationLectureDeLettre, activationLecture)
                    timer.start()


    # Affichage de l'image capturée par la webcam
    cv2.imshow("IMAGE", frame)
    cv2.imshow("HSV", tsv)
    cv2.imshow("GRIS", gris)
    cv2.imshow("CANNY", contours_canny)

    # Condition de sortie de la boucle While
    # > Touche Escape pour quitter
    key = cv2.waitKey(1)
    if key == 27:
        break

#On libère la webCam et on détruit toutes les fenêtres
webCam.release()
cv2.destroyAllWindows()