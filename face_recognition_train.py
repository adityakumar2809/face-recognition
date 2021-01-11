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
        minNeighbors=4
    )
    return faces_rect


def createTrainSet(people_names, pic_dir, cascade_classifier):
    features = []
    labels = []
    for person in people_names:
        path = os.path.join(pic_dir, person)
        label = people_names.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            bgr_image = cv2.imread(img_path)
            gray_image = getGrayscaleImage(bgr_image)

            faces_rect = detectFace(cascade_classifier, gray_image)
            for (x, y, w, h) in faces_rect:
                faces_roi = gray_image[y: y + h, x: x + w]
                features.append(faces_roi)
                labels.append(label)
    return features, labels


def createAndTrainFaceRecognizer(features, labels):
    features = np.array(features, dtype='object')
    labels = np.array(labels)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(features, labels)

    face_recognizer.save('trained_data/face_trained.yml')
    np.save('trained_data/features.npy', features)
    np.save('trained_data/labels.npy', labels)


def main():
    pic_dir = getPicDir()

    people_names = getFaceNames(pic_dir)
    print(people_names)

    haar_cascade = getCascadeClassifier()

    features, labels = createTrainSet(people_names, pic_dir, haar_cascade)
    print('==================TRAINING COMPLETED==================')
    print(f'Length of features = {len(features)}')
    print(f'Length of labels = {len(labels)}')
    print('======================================================')

    createAndTrainFaceRecognizer(features, labels)


if __name__ == "__main__":
    main()
