import numpy as np
import cv2

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('peopleCounter2.avi')
upper_body_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')

while(cap.isOpened()):
    ret, frame = cap.read() #read a frame
    upper_bodies = upper_body_cascade.detectMultiScale(frame)

    try:
        
        for (x, y, w, h) in upper_bodies:
            aux = cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 4)
            print('body found')
    except:
        # if there are no more frames to show...
        print('EOF')
        break
        
    cv2.imshow('Frame', frame)
    
    #Abort and exit with 'Q' or ESC
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release() #release video file
cv2.destroyAllWindows() #close all openCV windows
