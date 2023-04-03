import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def load_model(model_file, weights_file):
    return AccidentDetectionModel(model_file, weights_file)

def process_frame(frame, model):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(gray_frame, (250, 250))
    pred, prob = model.predict_accident(roi[np.newaxis, :, :])
    return pred, prob

def display_result(frame, pred, prob):
    if pred == "Accident":
        prob = round(prob[0][0] * 100, 2)
        cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"{pred} {prob}", (20, 30), font, 1, (255, 255, 0), 2)
    cv2.imshow('Video', frame)

def start_application():
    model = load_model("model.json", 'model_weights.h5')
    video = cv2.VideoCapture('cars.mp4')
    
    while True:
        ret, frame = video.read()
        
        if not ret:
            break  # Exit the loop if there are no more frames
        
        pred, prob = process_frame(frame, model)
        display_result(frame, pred, prob)
        
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break  # Exit the loop if 'q' is pressed
    
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_application()
