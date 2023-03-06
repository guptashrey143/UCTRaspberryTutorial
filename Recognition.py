import cv2
import pandas as pd
import datetime
import numpy as np
from PIL import Image
import os
from gpiozero import LED

green_led = LED(27)
red_led = LED(17)


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")

def user_name(user_id):
    df = pd.read_csv('data.csv')
    for i in df.index:
        if df['Id'][i] == user_id:
            name = df['Name'][i]
            return name
            
    return 'WRONG USER'
    
    
def recognizer_boundary(img, classifier, scaleFactor, minNeighbours, color, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbours)
    coords = []
    if len(coords) == 0:
        red_led.off()
        green_led.off()
    for (x, y, w, h)in features:
        cv2.rectangle(img, (x,y),(x+w,y+h), color, 2)
        user_id, _ = clf.predict(gray_img[y:y+h, x:x+w])
        name = user_name(user_id)
        if(name == 'WRONG USER'):
            red_led.on()
        else:
            green_led.on()
#         print(name)
        #name = get_name(user_id)
        cv2.putText(img, name, (x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x,y,w,h]
    return coords




def recognize(img, clf, faceCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "mix":(128,0,129)}
    coords = recognizer_boundary(img, faceCascade, 1.1, 10, color["green"], clf)
    
    if(len(coords) == 0):
        red_led.on()
    return img

VideoCapture = cv2.VideoCapture(0)

while True:
    _,img = VideoCapture.read()
    scale_percent = 10 # percent of original size
    width = int(img.shape[1] * scale_percent / 10)
    height = int(img.shape[0] * scale_percent / 10)
    dim = (width, height)
  
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img2 = recognize(img, clf, faceCascade)
    cv2.imshow("camera",img2)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

VideoCapture.release()        
cv2.destroyAllWindows()

green_led.off()
red_led.off()