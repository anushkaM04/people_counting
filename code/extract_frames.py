import cv2
import os
from tqdm import tqdm

video_path = "C:\\Users\\Spanidea-LT06\\Downloads\\myPOC\\3.mp4"          # video in same folder
output_folder = "C:\\Users\\Spanidea-LT06\\Downloads\\myPOC\\raw_fishes\\3.output"    # folder will be created

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Video FPS: {fps}")

saved = 0
with tqdm(desc="Extracting frames") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(
            output_folder, f"frame_{saved:05d}.jpg"
        )
        cv2.imwrite(frame_filename, frame)

        saved += 1
        pbar.update(1)

cap.release()
print(f"✅ Extraction complete. {saved} frames saved to '{output_folder}'.")