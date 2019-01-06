from model import get_inception_score
import os
import cv2

def test_inception(filepath="data/"):
    images = []
    for filename in os.listdir(filepath):
        img = cv2.imread(filepath + filename)
        images.append(img)
    print("NUM IMGS", len(images))
    inception_mean, inception_std = get_inception_score(images)
    print("MEAN: ", inception_mean)
    print("STD: ", inception_std)
