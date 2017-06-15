import numpy as np
import cv2

upper_body_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')

img = cv2.imread('images.jpg')

upper_bodies = upper_body_cascade.detectMultiScale(img)
for (x, y, w, h) in upper_bodies:
    aux = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 4)
    print('body found')

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()