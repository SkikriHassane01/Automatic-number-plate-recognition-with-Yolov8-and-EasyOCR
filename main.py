# ================================================>>
# ================================================>>
# _______________Table of Content:_________________
"""
1. Import libraries
2. Load the models
3. load the video or webcam
4. read the frames 
5. detect the vehicles
6. track it using Sort 
7. detect license plates
8. ...
"""
# =================================================<<
# =================================================<<


# _______________1. Import libraries_______________
from logger import logger
from ultralytics import YOLO
import cv2

# _______________2. load the models_______________
yolo_model = YOLO("./models/yolov8n.pt")
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# _______________3. load the video or webcam_______________
path_video = r"./Video/video1.mp4"
cap = cv2.VideoCapture(path_video)

num_frame = -1
vehicles = [2, 3, 5, 7] #id of the vehicles that we want to detect

# _______________4. read the frames_______________
while True and num_frame < 10:
    success, img = cap.read()

    if not success:
        print("Failed to read the frame or end of the video reached.")
