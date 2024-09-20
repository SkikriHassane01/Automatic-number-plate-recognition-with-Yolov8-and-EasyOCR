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

