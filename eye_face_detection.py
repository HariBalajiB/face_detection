# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:28:41 2020

@author: HARI
"""

import numpy as np
import cv2
import os

cam = cv2.VideoCapture(0)   
  
try: 
      
    # creating a folder named data 
    if not os.path.exists('data'): 
        os.makedirs('data') 
  
# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data')   
# frame 
currentframe = 0
n = 0
while(n<10): 
    # reading from frame 
    ret,frame = cam.read() 
    if ret: 
        # if video is still left continue creating images 
        name = 'C:/Users/HARI/data/frame' + str(currentframe) + '.jpg'
        print ('Creating...' + name) 
        # writing the extracted images 
        cv2.imwrite(name, frame) 
        # increasing counter so that it will 
        # show how many frames are created 
        currentframe += 1
    else: 
        break
    n=n+1
    
    if cv2.waitKey(1) == ord('q'):
        break
# Release all space and windows once done 
cam.release() 

face_cascade = cv2.CascadeClassifier('C:/Users/HARI/Desktop/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('C:/Users/HARI/Desktop/haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('C:/Users/HARI/Desktop/haarcascade_smile.xml')

img = cv2.imread('C:/Users/HARI/data/frame0.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#faces = face_cascade.detectMultiScale(gray)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    
#    eyes = eye_cascade.detectMultiScale(roi_gray)
#    for (ex,ey,ew,eh) in eyes:
#        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#        
#    smiling = smile_cascade.detectMultiScale(roi_gray, 1.7, 5)
#    for (sx,sy,sw,sh) in smiling:
#        cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(240,19,93),2)

cv2.imshow('img',img)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()