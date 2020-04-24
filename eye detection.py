#!/usr/bin/env python
# coding: utf-8



import numpy as np
import cv2

reye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
leye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
count = 0
cap = cv2.VideoCapture(0)

def eye_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    eyes = reye.detectMultiScale(img, 1.3, 5)
    eye = leye.detectMultiScale(img, 1.3, 5)
    if eyes is ():
        return None
    for (x,y,w,h) in eye:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    for (x,y,w,h) in eyes:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cropped_eye = img[y:y+h, x:x+w]
    return cropped_eye
while(True):
    ret, img = cap.read()
    gray = eye_extractor(img)
    if gray is not None:
        count += 1      
        eye = cv2.resize(gray, (200, 200))
        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
#       to store the eye samples        
        cv2.imwrite('path_to_folder' + str(count) + '.jpg', eye)
        cv2.putText(img, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('img',img)
    else:
        print("Eye not found")
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



