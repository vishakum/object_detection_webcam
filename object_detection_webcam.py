from ultralytics import YOLO
import cv2
import pyttsx3
import cvzone
import math



# initializing TTS(text to speach) engine
engine = pyttsx3.init()
engine.setProperty('rate', 90)
engine.setProperty('volume', 0.7)  # set volume


# function to speak detected objects
def speak(text):
    engine.say(text)
    engine.runAndWait()


# load YOLO model
model = YOLO('yolov8n.pt')
classnames = ["Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck",
              "Boat", "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking Meter",
              "Bench", "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear",
              "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase",
              "Frisbee", "Skis", "Snowboard", "Sports Ball", "Kite", "Baseball Bat",
              "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket", "Bottle",
              "Wine Glass", "Cup", "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple",
              "Sandwich", "Orange", "Broccoli", "Carrot", "Hot Dog", "Pizza", "Donut",
              "Cake", "Chair", "Couch", "Potted Plant", "Bed", "Dining Table", "Toilet",
              "TV", "Laptop", "Mouse", "Remote", "Keyboard", "Cell Phone", "Microwave",
              "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase", "Scissors"
    , "Teddy Bear", "Hair Drier", "Toothbrush",
              ]
# initialize webcam
cap = cv2.VideoCapture(0)  # for webcam
cap.set(3, 640)
cap.set(4, 480)

while True:
    # Loop for reading webcam imgs
    ret, img = cap.read()
    # perform image detection
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # for opencv bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # for cvzone rectangular border
            w, h = x2 - x1, y2 - y1
            # bbox=int(x1), int(y1), int(w), int(h) #converting weights in integer
            cvzone.cornerRect(img, (x1, y1, w, h))
            # confidence in finding object
            conf = math.ceil((box.conf[0] * 100)) / 100  # for show confidence in form of percentage
            print(conf)
            # class name
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classnames[cls]},{conf}', (max(0, x1 + 6), max(0, y1 - 10)))

            # Make the engine speak the detected class name and confidence
            speak(f"{classnames[cls]} ")

    # for open webcam
    cv2.imshow("image", img)
    cv2.waitKey(1)
