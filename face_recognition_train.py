import os
import cv2
import numpy as np


def getPicDir():
    cwd = os.path.dirname(__file__)
    pic_dir = os.path.join(cwd, 'images\\faces\\train')
    return pic_dir


def getFaceNames(pic_dir):
    people_names = []
    for name in os.listdir(pic_dir):
        people_names.append(name)
    return people_names


def getCascadeClassifier():
    haar_cascade = cv2.CascadeClassifier('haar_face.xml')
    return haar_cascade


def getGrayscaleImage(img):
    gray_image = cv2.cvtColor(
        src=img,
        code=cv2.COLOR_BGR2GRAY
    )
    return gray_image


def detectFace(cascade_classifier, img):
    faces_rect = cascade_classifier.detectMultiScale(
        image=img,
        scaleFactor=1.1,
        minNeighbors=10
    )
    return faces_rect


def getCroppedFaces(img, faces_rect):
    for (x, y, w, h) in faces_rect:
        faces_roi = img[y: y + h, x: x + w]
    return faces_roi


def main():
    pic_dir = getPicDir()

    people_names = getFaceNames(pic_dir)
    print(people_names)

    haar_cascade = getCascadeClassifier()


if __name__ == "__main__":
    main()
