import cv2
import numpy as np

face_csc = cv2.CascadeClassifier("C:\Users\oguzhan\Downloads\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml")
eyes_csc=cv2.CascadeClassifier("C:\Users\oguzhan\Downloads\opencv\sources\data\haarcascades\haarcascade_eye.xml")
cam=cv2.VideoCapture(1)

while(True):
    tf, img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_csc.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyes=eyes_csc.detectMultiScale(roi_gray)
    for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),3)
        
        

    cv2.imshow('img',img)
    key=cv2.waitKey(1)
    if key==27:
        break

cam.release()
cv2.destroyAllWindows()
