# For checking normally inferencing using locally saved weights, without in-out people counting logic

from ultralytics import YOLO
import cv2
import os

WEIGHTS_PATH = "C:\\Users\\Spanidea-LT06\\Downloads\\myPOC\\weights\\best3.pt"
IMAGE_DIR = "C:\\Users\\Spanidea-LT06\\Downloads\\myPOC\\test_images_for_inference"
VIDEO_PATH = "C:\\Users\\Spanidea-LT06\\Downloads\\myPOC\\New.mp4"
OUTPUT_DIR = "C:\\Users\\Spanidea-LT06\\Downloads\\myPOC\\Outputs_from_inference\\From_best3.pt_weights"

CONF_TRES = 0.25
IOU_THRES = 0.55

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load the YOLO model
model = YOLO(WEIGHTS_PATH)

# Perform inference on an IMAGES
print("Running inference on images...")

for img_name in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, img_name)

    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    results = model(
        img_path,
        conf=CONF_TRES,
        iou=IOU_THRES,
        show=True,
        project=OUTPUT_DIR,
        name="images_inference",
        exist_ok=True,
    )

print("Image inference completed.")

# Perform inference on a VIDEO
print("Running inference on video...")

model(
    VIDEO_PATH,
    conf=CONF_TRES,
    iou=IOU_THRES,
    show=True,
    project=OUTPUT_DIR,
    name="video_inference",
    exist_ok=True,
)

print("Video inference completed.")

for r in results:
    boxes = r.boxes.xyxy
    scores = r.boxes.conf
    classes = r.boxes.cls