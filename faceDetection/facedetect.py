import cv2
import matplotlib as plt
import numpy as np


# Training data for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Image to detect face on
img = cv2.imread('family3.jpeg')
imS = cv2.resize(img, (960, 540))

# converting image to grayscale
gray = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)

# detect faces in grayscale image using face_cascade data
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# draw rectangle around faces
for (x,y,w,h) in faces:
    cv2.rectangle(imS, (x,y), (x+w, y+h), (255, 0, 0), 2)

# show resulting image



