import cv2
import numpy as np


def getCascadeClassifier():
    haar_cascade = cv2.CascadeClassifier('haar_face.xml')
    return haar_cascade


def main():
    haar_cascade = getCascadeClassifier()


if __name__ == "__main__":
    main()
