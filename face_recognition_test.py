import cv2
import os
import numpy as np


def getCascadeClassifier():
    haar_cascade = cv2.CascadeClassifier('haar_face.xml')
    return haar_cascade


def loadFeaturesAndLabels():
    features = np.load('trained_data/features.npy', allow_pickle=True)
    labels = np.load('trained_data/labels.npy', allow_pickle=True)
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


def recognize_face(img, gray_img,
                   cascade_classifier,
                   face_recognizer,
                   people_names):
    faces_rect = cascade_classifier.detectMultiScale(
        image=gray_img,
        scaleFactor=1.1,
        minNeighbors=4
    )

    for (x, y, w, h) in faces_rect:
        faces_roi = gray_img[y: y + h, x: x + w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(
            f'Label = {people_names[label]} with a confidence of {confidence}'
        )

        cv2.putText(
            img=img,
            text=str(people_names[label]),
            org=(20, 20),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=1.0,
            color=(0, 255, 0),
            thickness=2
        )

        cv2.rectangle(
            img=img,
            pt1=(x, y),
            pt2=(x + w, y + h),
            color=(0, 255, 0),
            thickness=2
        )

    return img


def main():
    haar_cascade = getCascadeClassifier()
    # features, labels - loadFeaturesAndLabels()

    face_recognizer = createAndReadFaceRecognizer()

    pic_dir = getPicDir()
    people_names = getFaceNames(pic_dir)

    img = cv2.imread('images/faces/val/ben_afflek/2.jpg')
    gray_image = getGrayscaleImage(img)

    cv2.imshow('Unidentified Person', img)

    detected_image = recognize_face(
        img,
        gray_image,
        haar_cascade,
        face_recognizer,
        people_names
    )
    cv2.imshow('Identified Person', detected_image)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
