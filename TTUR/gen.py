import numpy as np
import cv2

for i in range(100):
    img = np.random.randint(255, size=(20, 20, 3))
    cv2.imwrite("img" + str(i) + ".jpg", img)



