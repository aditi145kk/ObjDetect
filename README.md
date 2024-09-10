PPE Detection with YOLOv8
This project uses a YOLOv8n model to detect Personal Protective Equipment (PPE) in images. It leverages OpenCV for image processing and visualization.

Overview
The model detects PPE equipment in images and draws bounding boxes around detected objects. This script was trained with 100 epochs on a dataset of 1000 images. For improved results, consider training with a larger dataset and more epochs.

Features
Uses YOLOv8n model for object detection.
Supports detection of various classes such as Person, Car, Truck, Van, and Vehicle.
Processes images from a specified input directory.
Saves annotated images with bounding boxes and confidence scores to a specified output directory.
Requirements
Ensure you have the following libraries installed:

ultralytics (for YOLOv8 model)
opencv-python (for image processing)
cvzone (for easy drawing functions)
