# YOLO Model Object Detection Classes

This README provides a list of all the object classes detected by the YOLO model along with their corresponding indices.

## Table of Contents:
1. [Introduction](#introduction)
2. [Object Classes and Indices](#object-classes-and-indices)
3. [How to Use the Model](#how-to-use-the-model)

---

## Introduction

This model is based on the YOLOv8 architecture. The model is trained on the COCO dataset, which contains 80 object classes. Each class has an associated index, which is used during the object detection process. 

The list below includes all object classes and their indices.

---

## Object Classes and Indices

| Index | Object Class         |
|-------|----------------------|
| 0     | Person               |
| 1     | Bicycle              |
| 2     | Car                  |
| 3     | Motorcycle           |
| 4     | Airplane             |
| 5     | Bus                  |
| 6     | Train                |
| 7     | Truck                |
| 8     | Boat                 |
| 9     | Traffic Light        |
| 10    | Fire Hydrant         |
| 11    | Stop Sign            |
| 12    | Parking Meter        |
| 13    | Bench                |
| 14    | Bird                 |
| 15    | Cat                  |
| 16    | Dog                  |
| 17    | Horse                |
| 18    | Sheep                |
| 19    | Cow                  |
| 20    | Elephant             |
| 21    | Bear                 |
| 22    | Zebra                |
| 23    | Giraffe              |
| 24    | Backpack             |
| 25    | Umbrella             |
| 26    | Handbag              |
| 27    | Tie                  |
| 28    | Suitcase             |
| 29    | Frisbee              |
| 30    | Skis                 |
| 31    | Snowboard            |
| 32    | Sports Ball          |
| 33    | Kite                 |
| 34    | Baseball Bat         |
| 35    | Baseball Glove       |
| 36    | Skateboard           |
| 37    | Surfboard            |
| 38    | Tennis Racket        |
| 39    | Bottle               |
| 40    | Wine Glass           |
| 41    | Cup                  |
| 42    | Fork                 |
| 43    | Knife                |
| 44    | Spoon                |
| 45    | Bowl                 |
| 46    | Banana               |
| 47    | Apple                |
| 48    | Sandwich             |
| 49    | Orange               |
| 50    | Broccoli             |
| 51    | Carrot               |
| 52    | Hot Dog              |
| 53    | Pizza                |
| 54    | Donut                |
| 55    | Cake                 |
| 56    | Chair                |
| 57    | Couch                |
| 58    | Potted Plant         |
| 59    | Bed                  |
| 60    | Dining Table         |
| 61    | Toilet               |
| 62    | TV                   |
| 63    | Laptop               |
| 64    | Mouse                |
| 65    | Remote               |
| 66    | Keyboard             |
| 67    | Cell Phone           |
| 68    | Microwave            |
| 69    | Oven                 |
| 70    | Toaster              |
| 71    | Sink                 |
| 72    | Refrigerator         |
| 73    | Book                 |
| 74    | Clock                |
| 75    | Vase                 |
| 76    | Scissors             |
| 77    | Teddy Bear           |
| 78    | Hair Drier           |
| 79    | Toothbrush           |

---

## How to Use the Model

To use this YOLO model for object detection, follow the steps below:

1. **Import YOLO library**:

```python
from ultralytics import YOLO
```

2. **Load the pre-trained model**

```python 
model = YOLO('./models/yolov8n.pt')
```