from imutils.video import FileVideoStream
import cv2
import time 
from imutils.video import VideoStream
import imutils
import os
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import statistics 
from keras.models import load_model
from mtcnn.mtcnn import MTCNN

print("Loading model...")
facenet_model = load_model('./model/facenet_keras.h5')


def getLabel(face_image):
    face_array = np.asarray(face_image, dtype='float32')
    mean, std = face_array.mean(), face_array.std()
    face_array = (face_array - mean) / std 
    yhat = facenet_model.predict(np.expand_dims(face_array, axis=0))

    distances = []
    for train_vec, train_truth_label, _ in all_data:
        dist = np.linalg.norm(yhat - train_vec)
        distances.append((dist, train_truth_label))

    distances.sort()
    
    #print(distances[:5])
    #print()
    #print()

    if distances[0][0] > 9.5:
        return "UNREGISTERED"

    try:
        test_predicted_label = statistics.mode([l for _,l in distances[:5]])
        cs = 0
        ms = 0
        c = 0
        for d, l in distances[:5]:
            if l != test_predicted_label:
                ms += d
                c += 1
            else:
                cs += d
        if c > 0:
            print(test_predicted_label, (ms/c) - (cs/(5-c)), c)
            if c > 1:
                return "UNREGISTERED"
        #if c < 5:

    except statistics.StatisticsError as e:
        test_predicted_label = 'UNREGISTERED'
    
    
    
    return test_predicted_label


def process(frame, required_size=(160, 160)):
    detector = MTCNN()
    width, height = frame.size
    pixels = np.asarray(frame)
    drawing = ImageDraw.Draw(frame)
    results = detector.detect_faces(pixels)
    #print([res['confidence'] for res in results])

    for face in results:
        if face['confidence'] < 0.85:
            continue
        x1, y1, l, w = face['box']
        x2, y2 = x1 + l, y1 + w
        x1 = max(min(x1, width), 0)
        x2 = max(min(x2, width), 0)
        y1 = max(min(y1, height), 0)
        y2 = max(min(y2, height), 0)

        drawing.rectangle(((x1,y1),(x2,y2)), fill=None, width=2, outline='yellow')
        
        image = Image.fromarray(pixels[y1:y2, x1:x2]).resize(required_size)
        label = getLabel(image)
        drawing.text( (x1,y1-15), label)
    
    return frame


print("Loading training data...")
all_data = []

for personName in os.listdir('vectors'):
    for filename in os.listdir('vectors/' + personName):
        vec = np.load('vectors/' + personName + '/' + filename)
        all_data.append((vec, personName, 'dataset/' + personName + '/' + filename[:-4]))

print("Starting video stream...")
"""
vs = FileVideoStream('output.avi').start()
fileStream = True


frameIndex = 0

while True:
    if fileStream and not vs.more():
        break
    pixels = vs.read()
    frameIndex += 1    
    #print(frameIndex)
    
    if frameIndex % 30 == 0:
        pixels = imutils.resize(pixels, width=900)
        frame = Image.fromarray(pixels)

        #print(type(frame))
        frame = process(frame)
        pixels = np.asarray(frame)
        cv2.imshow("Frame", pixels)
        cv2.waitKey(1)
"""
frameIndex = 0
cap = cv2.VideoCapture('rtsp://admin:12345@192.168.1.160:8554')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 25.0, (1920,1080))
while(cap.isOpened()):
    ret, pixels = cap.read()
    frameIndex += 1 
    if ret==True:
        if frameIndex % 30 == 0:
            pixels = imutils.resize(pixels, width=900)
            frame = Image.fromarray(pixels)

            #print(type(frame))
            frame = process(frame)
            pixels = np.asarray(frame)
            cv2.imshow("Frame", pixels)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

    