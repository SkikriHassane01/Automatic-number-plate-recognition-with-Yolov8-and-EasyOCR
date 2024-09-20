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
from sort import *
# _______________2. load the models_______________
yolo_model = YOLO("./models/yolov8n.pt")
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# _______________3. load the video or webcam_______________
path_video = r"./Video/video1.mp4"
cap = cv2.VideoCapture(path_video)

num_frame = -1
vehicles = [2, 3, 5, 7] # id of the vehicles that we want to detect
tracker = Sort()

# _______________4. read the frames_______________
while True and num_frame < 10:
    success, img = cap.read()

    if not success:
        print("Failed to read the frame or end of the video reached.")

    # _______________5. detect the vehicles_______________
    detections = yolo_model(img)[0]
    """
    detections contains the predicted bounding boxes, class labels,
    confidence scores, and other information for each detected object
    in the image.
    """
    detections_bbox_score = []
    # logger.info(detections)

    for detection in detections.boxes.data.tolist():
        # logger.info(detection)
        x1, y1, x2, y2, score, class_id = detection
        if float(class_id) in vehicles:
            detections_bbox_score.append([x1, y1, x2, y2, score])


    # _______________5. Track vehicles_______________

    tracker_ids = tracker.update(np.asarray(detections_bbox_score))
    """
    Params of the update function:
        a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.
    """
    
    cv2.imshow("License Plate Detector", img)
        

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
