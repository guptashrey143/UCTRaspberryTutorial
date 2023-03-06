import cv2
import pandas as pd
import numpy as np
from PIL import Image
import os
from gpiozero import LED

green_led = LED(27)
red_led = LED(17)


df = pd.read_csv('data.csv')


students_registered = df.shape[0]


## Calling the Classifiers

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

## Function to check if a given frame contains a face or not and if it does add it to the dataset

def generator(img, faceCascade, img_id, user_id):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "mix":(128,0,129)}
    coords = detector(img,faceCascade, 1.1, 10, color['green'], 'face')
    
    if len(coords) == 4:
        red_led.off()
        green_led.on()
        roi_img = img[coords[1]: coords[1] + coords[3], coords[0]: coords[0] + coords[3]]
        dataset_generator(roi_img,user_id, img_id)
    else:
        green_led.off()
        red_led.on()
    return img

##Function to store the image in the dataset

def dataset_generator(img, id, img_id):
    cv2.imwrite("/home/pi/Python Projects/Images/user." + str(id) + "." + str(img_id) + ".jpg", img)


##Function to find the image on basis of classifier and return its coordinates

def detector(img, classifier, scaleFactor, minNeighbours, color,txt):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbours)
    coords = []
    for (x, y, w, h)in features:
        cv2.rectangle(img, (x,y),(x+w,y+h), color, 2)
        cv2.putText(img, txt, (x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x,y,w,h]
    return coords

## Function to take the details of students from the user

def get_data():
    enroll = input("Enter ID.:")
    name = input("Enter Name:")
    return enroll,name

## Function to write the details of the students in a csv file

def write_data(enroll, name, i):
    if i > 0:
        df = pd.read_csv('data.csv')
        df = df.iloc[:,1:]
        df = df.append(dict(zip(df.columns,[enroll, name])), ignore_index=True)
        df.to_csv('data.csv')
        
    else:
        data = {'Id': [enroll],'Name': [name]}
        df = pd.DataFrame(data)
        df.to_csv('data.csv')


## Creating the dataset of the images of students when they get registered

VideoCapture = cv2.VideoCapture(0)

user_id,name = get_data()
write_data(user_id,name,students_registered)
students_registered+=1
img_id = 0
while True:
    _,img = VideoCapture.read()
    
    scale_percent = 10 # percent of original size
    width = int(img.shape[1] * scale_percent / 10)
    height = int(img.shape[0] * scale_percent / 10)
    dim = (width, height)
  
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    img2 = generator(img, faceCascade,img_id,user_id)

    img_id+=1
    cv2.imshow("camera",img2)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

VideoCapture.release()
cv2.destroyAllWindows()
red_led.off()
green_led.off()


from PIL import Image
import os

def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []
    
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
        
        
        faces.append(imageNp)
        ids.append(id)
        
    ids = np.array(ids)
    print(ids)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.yml")
    
    
    
train_classifier('/home/pi/Python Projects/Images')
