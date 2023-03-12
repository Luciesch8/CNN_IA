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

#dimensions de l'ardoise
zoneEcritureLongueurMin = 540
zoneEcritureLongueurMax = 590
zoneEcritureLargeurMin = 300
zoneEcritureLargeurMax = 340


print('Initialisation de la webcam')
webCam = cv2.VideoCapture(0)
if webCam.isOpened():
    longueurWebcam = webCam.get(3)
    largeurWebcam = webCam.get(4)
    print('Résolution:' + str(longueurWebcam) + " X " + str(largeurWebcam))
else:
    print('ERREUR')

while True:

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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)


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