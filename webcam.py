import numpy as np
import cv2 as cv
import tensorflow as tf
import utils
from keras import models

cap = cv.VideoCapture(0)
model = models.load_model("checkpoints/model1.keras")
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
upleft = int(width / 2 - height / 2)
upright = int(width / 2 + height / 2)
kernel = np.ones((1, 1), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_resized = frame[0:height, upleft:upright]
    frame_resized = cv.resize(frame_resized, (128, 128))
    frame_resized = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)
    _, frame_resized = cv.threshold(frame_resized, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    frame_resized = cv.dilate(frame_resized, kernel, iterations=1)
    # frame_resized = utils.fourier(frame_resized)
    frame_resized = cv.cvtColor(frame_resized, cv.COLOR_GRAY2BGR)
    frame_resized = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)
    frame_resized = cv.cvtColor(frame_resized, cv.COLOR_GRAY2BGR)
    cv.imshow('Webcam', frame_resized)
    frame_resized = np.expand_dims(frame_resized, axis=0)  # Add batch dimension
    # frame_resized = frame_resized.astype(np.float32) / 255.0
    # model.predict(frame_resized)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
