from ultralytics import YOLO
import cv2
import cvzone
import math
import os

'''
This Project Uses a YOLOv8n model to detect PPE Equipment in images using OpenCV.
This model uses 100 epochs and was trained on a dataset of 1000 images. 
It is recommended to train on more images and more epochs for better results.
'''

# Set the model and class names
model = YOLO("yolov8n.pt")

classNames = [
     'Person', 'car', 'truck', 'van', 'vehicle'
]

# Directory with images to process
input_dir = r"C:\Users\suvar\OneDrive\Desktop\Obj-Detect\input" 
output_dir = r"C:\Users\suvar\OneDrive\Desktop\Obj-Detect\output"


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image in the directory
for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error reading image {image_path}")
        continue
    
    # Perform detection
    results = model(img)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9)  
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), thickness=2)
    
   
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, img)

 
# Cleanup
cv2.destroyAllWindows()
