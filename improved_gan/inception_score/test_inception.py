from .model import get_inception_score
import os
import cv2

def test_inception(filepath="data/"):
    images = []
    for filename in os.listdir(filepath):
        img = cv2.imread(os.path.join(filepath, filename))
        images.append(img)
    print("NUM IMGS", len(images))
    print("image shape", images[0].shape)
    inception_mean, inception_std = get_inception_score(images)
    print("\n")
    print("MEAN Incepton Score: ", inception_mean)
    print("STD Inception Score: ", inception_std)
    return inception_mean, inception_std
