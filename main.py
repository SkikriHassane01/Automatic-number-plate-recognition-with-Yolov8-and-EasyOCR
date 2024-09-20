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
from utils import get_car, read_license_plate, write_result_csv_file

# _______________2. load the models_______________
yolo_model = YOLO("./models/yolov8n.pt")
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# _______________3. load the video or webcam_______________
path_video = r"./Video/video1.mp4"
cap = cv2.VideoCapture(path_video)

frame_number = -1
vehicles = [2, 3, 5, 7] # id of the vehicles that we want to detect
tracker = Sort()
results = {}

# _______________4. read the frames_______________
while True:
    success, img = cap.read()

    if not success:
        print("Failed to read the frame or end of the video reached.")
        break  # Break the loop if the video ends or fails

    frame_number +=1
    results[frame_number] = {}

    # _______________5. detect the vehicles_______________
    detections = yolo_model(img)[0]
    detections_bbox_score = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_bbox_score.append([x1, y1, x2, y2, score])

    if len(detections_bbox_score) == 0:
        continue  # Skip frame if no vehicles detected

    # _______________6. Track vehicles_______________
    tracker_ids = tracker.update(np.asarray(detections_bbox_score))

    # _______________7. detect license plates_______________
    license_plates_detections = license_plate_detector(img)[0]

    if len(license_plates_detections.boxes.data.tolist()) == 0:
        continue  # Skip if no license plates detected

    for license_detected in license_plates_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_detected

        # _______________8. assign license plate to their car_______________
        x1car, y1car, x2car, y2car, car_id = get_car(license_detected, tracker_ids)

        if car_id != -1:
            # _______________9. Crop and process the license plate_______________
            license_detected_crop = img[int(y1):int(y2), int(x1):int(x2), :]
            license_detected_gray = cv2.cvtColor(license_detected_crop, cv2.COLOR_BGR2GRAY)
            _, license_detected_thresh = cv2.threshold(license_detected_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # _______________10. Read license plate number_______________
            license_plate_text, license_plate_confidence = read_license_plate(license_detected_crop)

            # _______________11. Results_______________
            if license_plate_text is not None:
                results[frame_number][car_id] = {'car': {'bbox': [x1car, y1car, x2car, y2car]},
                                                 'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                   'text': license_plate_text,
                                                                   'bbox_score': score,
                                                                   'text_score': license_plate_confidence}
                                                 }

    cv2.imshow("License Plate Detector", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

write_result_csv_file(results, './results.csv')