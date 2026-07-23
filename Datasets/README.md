# Evaluation Dataset

This directory contains the sample images used to evaluate the perception stage of the VIGIL pipeline.

## Dataset Description

The dataset consists of **9 public real-world images** covering a variety of everyday scenarios, including:

- Traffic intersections
- Parking areas
- Pedestrians
- Cars and motorcycles
- Bicycles
- Indoor office environments

These images were selected to validate the object detection capabilities of the project across different scene types.

## Evaluation Method

The images were processed using the **pretrained Ultralytics YOLOv8n** object detection model.

For each image, the model generated object detections along with confidence scores and bounding boxes. The prediction outputs were visually inspected to verify the correctness of the detected objects.

## Purpose

This dataset was used to:

- Validate the perception stage of the VIGIL pipeline.
- Verify that common real-world objects are detected correctly.
- Measure inference time across different scenes.
- Demonstrate the object detection workflow before downstream reasoning.

## Contents

```
Datasets/
├── README.md
└── images/
    ├── Screenshot 2026-07-22 200844.png
    ├── Screenshot 2026-07-22 201046.png
    ├── ...
```

## Summary

- **Images evaluated:** 9
- **Detection model:** Ultralytics YOLOv8n (pretrained)
- **Average inference time:** ~83 ms per image
- **Primary object classes detected:** Person, Car, Bicycle, Bus, Motorcycle, Truck, Laptop, Chair, Book, Umbrella, Pizza, Dining Table, Couch, Cup, Tie

This dataset is included for evaluation purposes and demonstrates the object detection stage of the VIGIL pipeline.