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
8. assign license plate to their car
"""
# =================================================<<
# =================================================<<


# _______________1. Import libraries_______________
from logger import logger
from ultralytics import YOLO
import cv2
from sort import *
from utils import get_car, read_license_plate

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


    # _______________6. Track vehicles_______________

    tracker_ids = tracker.update(np.asarray(detections_bbox_score))
    """
    Params of the update function:
        a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.
    """
    
    # _______________7. detect license plates_______________
    license_plates_detections = license_plate_detector(img)[0]
    
    for license_detected in license_plates_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_detected
    
        # _______________8. assign license plate to their car_______________
        x1car, y1car, x2car, y2car, car_id = get_car(license_detected, tracker_ids)
        
        # _______________9. Crop and process the license plate_______________
        license_detected_crop = img[int(y1) : int(y2), int(x1) : int(x2), :] 
        license_detected_gray = cv2.cvtColor(license_detected_crop,cv2.COLOR_BGR2GRAY)
        _, license_detected_thresh = cv2.threshold(license_detected_gray, 64, 255, cv2.THRESH_BINARY_INV) # Any pixel value in license_detected_gray below 64 will be set to the maximum value (255), while pixels above 64 will be set to 0
        
        # cv2.imshow("Cropped image", license_detected_crop)
        # cv2.imshow("thresh image", license_detected_thresh)
        # cv2.waitKey(0)
        
        # _______________10. Read license plate number_______________
        license_plate_text, license_plate_confidence = read_license_plate(license_detected_crop)
        
        
        # _______________11. Results_______________
        
        
        
        
        
    # cv2.imshow("License Plate Detector", img)
        

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
