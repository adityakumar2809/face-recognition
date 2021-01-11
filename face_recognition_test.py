import cv2
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


def main():
    haar_cascade = getCascadeClassifier()
    features, labels - loadFeaturesAndLabels()

    face_recognizer = createAndReadFaceRecognizer()


if __name__ == "__main__":
    main()
