import os
import cv2
import numpy as np


def getFaceNames():
    people_names = []
    cwd = os.path.dirname(__file__)
    pic_dir = os.path.join(cwd, 'images\\faces\\train')
    for name in os.listdir(pic_dir):
        people_names.append(name)
    return people_names


def main():
    people_names = getFaceNames()
    print(people_names)


if __name__ == "__main__":
    main()
