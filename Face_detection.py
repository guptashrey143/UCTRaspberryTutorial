import cv2
from gpiozero import LED



green_led = LED(27)
red_led = LED(17)

green_led.off()
red_led.off()


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detector(img, classifier, scaleFactor, minNeighbours, color,txt):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbours)
    coords = []
    for (x, y, w, h)in features:
        cv2.rectangle(img, (x,y),(x+w,y+h), color, 2)
        cv2.putText(img, txt, (x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x,y,w,h]
    return coords

def generator(img, faceCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "mix":(128,0,129)}
    coords = detector(img,faceCascade, 1.1, 10, color['green'], 'face')
    if(len(coords) != 0):
        green_led.on()
        red_led.off()
    else:
        green_led.off()
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
    img2 = generator(img, faceCascade)
    cv2.imshow("camera",img2)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

VideoCapture.release()
cv2.destroyAllWindows()
green_led.off()
red_led.off()
