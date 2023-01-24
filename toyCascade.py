import cv2
import numpy as np
import time

toyCascade = cv2.CascadeClassifier('myhaar.xml')

cap = cv2.VideoCapture(0)

prevFrameTime = 0
newFrameTime = 0 

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    toy = toyCascade.detectMultiScale(gray, 1.1, 2)

    print(toy)

    for (x,y,w,h) in toy:
        cv2.rectangle(img, (x, y), (x+200, y+200), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    img = cv2.resize(img, (500, 300))
    font = cv2.FONT_HERSHEY_SIMPLEX
    newFrameTime = time.time()
    fps = 1/(newFrameTime - prevFrameTime)
    prevFrameTime = newFrameTime

    fps = int(fps)
    fps = str(fps)

    cv2.putText(img, fps, (7, 70), font, 3, (100, 0, 100), 3, cv2.LINE_AA)    
        
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()