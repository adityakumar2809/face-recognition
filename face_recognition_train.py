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


def main():
    pic_dir = getPicDir()
    people_names = getFaceNames(pic_dir)
    print(people_names)


if __name__ == "__main__":
    main()
