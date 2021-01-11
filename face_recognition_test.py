import cv2
import os
import numpy as np


def getCascadeClassifier():
    haar_cascade = cv2.CascadeClassifier('haar_face.xml')
    return haar_cascade


def loadFeaturesAndLabels():
    features = np.load('trained_data/features.npy')
    labels = np.load('trained_data/labels.npy')
    return features, labels


def createAndReadFaceRecognizer():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('trained_data/face_trained.yml')
    return face_recognizer


def getPicDir():
    cwd = os.path.dirname(__file__)
    pic_dir = os.path.join(cwd, 'images\\faces\\train')
    return pic_dir


def getFaceNames(pic_dir):
    people_names = []
    for name in os.listdir(pic_dir):
        people_names.append(name)
    return people_names


def getGrayscaleImage(img):
    gray_image = cv2.cvtColor(
        src=img,
        code=cv2.COLOR_BGR2GRAY
    )
    return gray_image


def main():
    haar_cascade = getCascadeClassifier()
    features, labels - loadFeaturesAndLabels()

    face_recognizer = createAndReadFaceRecognizer()

    pic_dir = getPicDir()
    people_names = getFaceNames(pic_dir)

    img = cv2.imread(r'../images\faces\val\ben_afflek/1.jpg')
    gray_image = getGrayscaleImage(img)


if __name__ == "__main__":
    main()
