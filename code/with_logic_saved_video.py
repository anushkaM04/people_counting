# --- same code as below just added FPS Calculation overlay on saved video inference with main in/out logic----

from ultralytics import YOLO
import cv2
import time
import os

# ==========================
# CONFIG
# ==========================
WEIGHTS_PATH = r"C:\\Users\\Spanidea-LT06\\Downloads\\myPOC\\weights\\best.pt"
VIDEO_PATH = r"C:\\Users\\Spanidea-LT06\\Downloads\\myPOC\\New.mp4"
OUTPUT_PATH = r"C:\\Users\\Spanidea-LT06\\Downloads\\myPOC\\Outputs_from_inference\\From_best.pt_weights.mp4"

CONF_THRES = 0.70
IOU_THRES = 0.63
IMG_SIZE = 512

# Reference line
LINE_Y = IMG_SIZE // 2

# ==========================
# LOAD MODEL
# ==========================
model = YOLO(WEIGHTS_PATH)

# ==========================
# OPEN VIDEO
# ==========================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("❌ Could not open video file")

video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps == 0:
    video_fps = 25

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    video_fps,
    (IMG_SIZE, IMG_SIZE)
)

print("🎬 Video opened. Processing...")

# ==========================
# FPS CALCULATION STATE
# ==========================
fps_start_time = time.time()
fps_frame_count = 0
fps_display = 0.0
FPS_UPDATE_INTERVAL = 0.5  # seconds

# ==========================
# TRACKING STATE
# ==========================
track_last_side = {}
track_history_y = {}
track_last_frame = {}
track_counted = {}

MAX_TRACK_GAP = 15
CROSSING_COOLDOWN = 25
LINE_TOLERANCE = 5

in_count = 0
out_count = 0
frame_number = 0

# ==========================
# VIDEO LOOP
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    fps_frame_count += 1

    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    annotated_frame = frame_resized.copy()

    # YOLO tracking
    results = model.track(
        source=frame_resized,
        conf=CONF_THRES,
        iou=IOU_THRES,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # ==========================
    # PROCESS DETECTIONS
    # ==========================
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(boxes, track_ids, classes):
            if int(cls) != 0:
                continue

            x1, y1, x2, y2 = box
            center_y = int((y1 + y2) / 2)

            cv2.rectangle(annotated_frame,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 255, 0), 2)

            cv2.putText(annotated_frame,
                        f"ID {track_id}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            current_side = "above" if center_y < LINE_Y else "below"

            if track_id not in track_counted:
                track_counted[track_id] = {"IN": 0, "OUT": 0}

            if track_id in track_history_y:
                prev_y = track_history_y[track_id]
                prev_side = track_last_side.get(track_id)

                line_top = LINE_Y - LINE_TOLERANCE
                line_bottom = LINE_Y + LINE_TOLERANCE

                line_crossed = False
                crossing_type = None

                if prev_y < line_top and center_y > line_bottom:
                    line_crossed = True
                    crossing_type = "IN"
                elif prev_y > line_bottom and center_y < line_top:
                    line_crossed = True
                    crossing_type = "OUT"

                if not line_crossed and prev_side:
                    if prev_side == "above" and current_side == "below":
                        line_crossed = True
                        crossing_type = "IN"
                    elif prev_side == "below" and current_side == "above":
                        line_crossed = True
                        crossing_type = "OUT"

                if line_crossed:
                    last_counted = track_counted[track_id].get(crossing_type, 0)
                    if last_counted == 0 or frame_number - last_counted >= CROSSING_COOLDOWN:
                        if crossing_type == "IN":
                            in_count += 1
                            track_counted[track_id]["IN"] = frame_number
                        else:
                            out_count += 1
                            in_count = max(0, in_count - 1)
                            track_counted[track_id]["OUT"] = frame_number

            track_last_side[track_id] = current_side
            track_history_y[track_id] = center_y
            track_last_frame[track_id] = frame_number

    # ==========================
    # FPS UPDATE (SMOOTHED)
    # ==========================
    elapsed = time.time() - fps_start_time
    if elapsed >= FPS_UPDATE_INTERVAL:
        fps_display = fps_frame_count / elapsed
        fps_start_time = time.time()
        fps_frame_count = 0

    # ==========================
    # DRAW OVERLAYS
    # ==========================
    cv2.line(annotated_frame,
             (0, LINE_Y),
             (IMG_SIZE, LINE_Y),
             (0, 0, 255), 2)

    cv2.putText(annotated_frame,
                f"IN: {in_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.putText(annotated_frame,
                f"OUT: {out_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    cv2.putText(annotated_frame,
                f"FPS: {fps_display:.1f}",
                (IMG_SIZE - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 0), 2)

    writer.write(annotated_frame)
    cv2.imshow("YOLO Video | IN-OUT People Counting", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================
# CLEANUP
# ==========================
cap.release()
writer.release()
cv2.destroyAllWindows()
print("✅ Video processing completed.")
print(f"Saved output to: {OUTPUT_PATH}")





# # ------ After Edits ------

# from ultralytics import YOLO
# import cv2
# import time
# import os

# # ==========================
# # CONFIG
# # ==========================
# WEIGHTS_PATH = r"C:\\Users\\Spanidea-LT06\\Downloads\\myPOC\\weights\\best.pt"
# VIDEO_PATH = r"C:\\Users\\Spanidea-LT06\\Downloads\\myPOC\\New.mp4"
# OUTPUT_PATH = r"C:\\Users\\Spanidea-LT06\\Downloads\\myPOC\\Outputs_from_inference\\From_best.pt_weights"

# CONF_THRES = 0.70
# IOU_THRES = 0.63
# IMG_SIZE = 512

# # Reference line
# LINE_Y = IMG_SIZE // 2

# # ==========================
# # LOAD MODEL
# # ==========================
# model = YOLO(WEIGHTS_PATH)

# # ==========================
# # OPEN VIDEO
# # ==========================
# cap = cv2.VideoCapture(VIDEO_PATH)

# if not cap.isOpened():
#     raise RuntimeError("❌ Could not open video file")

# fps = cap.get(cv2.CAP_PROP_FPS)
# if fps == 0:
#     fps = 25  # fallback

# # Video writer
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# writer = cv2.VideoWriter(
#     OUTPUT_PATH,
#     fourcc,
#     fps,
#     (IMG_SIZE, IMG_SIZE)
# )

# print("🎬 Video opened. Processing...")

# # ==========================
# # TRACKING STATE (UNCHANGED)
# # ==========================
# track_last_side = {}
# track_history_y = {}
# track_first_seen = {}
# track_last_frame = {}
# track_counted = {}

# MAX_TRACK_GAP = 15
# CROSSING_COOLDOWN = 25
# LINE_TOLERANCE = 5

# in_count = 0
# out_count = 0
# frame_number = 0

# # ==========================
# # VIDEO LOOP
# # ==========================
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_number += 1
#     frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
#     annotated_frame = frame_resized.copy()

#     # YOLO tracking
#     results = model.track(
#         source=frame_resized,
#         conf=CONF_THRES,
#         iou=IOU_THRES,
#         persist=True,
#         tracker="bytetrack.yaml",
#         verbose=False
#     )

#     # ==========================
#     # PROCESS DETECTIONS
#     # ==========================
#     current_frame_track_ids = set()
#     previous_track_ids = set(track_last_side.keys())

#     if results[0].boxes.id is not None:
#         boxes = results[0].boxes.xyxy.cpu().numpy()
#         track_ids = results[0].boxes.id.cpu().numpy().astype(int)
#         classes = results[0].boxes.cls.cpu().numpy()

#         current_frame_track_ids = set(track_ids.tolist())

#         for box, track_id, cls in zip(boxes, track_ids, classes):
#             if int(cls) != 0:
#                 continue

#             x1, y1, x2, y2 = box
#             center_y = int((y1 + y2) / 2)

#             # Draw bbox
#             cv2.rectangle(
#                 annotated_frame,
#                 (int(x1), int(y1)),
#                 (int(x2), int(y2)),
#                 (0, 255, 0), 2
#             )

#             cv2.putText(
#                 annotated_frame,
#                 f"ID {track_id}",
#                 (int(x1), int(y1) - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6, (0, 255, 0), 2
#             )

#             current_side = "above" if center_y < LINE_Y else "below"

#             if track_id not in track_counted:
#                 track_counted[track_id] = {"IN": 0, "OUT": 0}

#             if track_id in track_history_y:
#                 prev_y = track_history_y[track_id]
#                 prev_side = track_last_side.get(track_id)

#                 line_top = LINE_Y - LINE_TOLERANCE
#                 line_bottom = LINE_Y + LINE_TOLERANCE

#                 line_crossed = False
#                 crossing_type = None

#                 if prev_y < line_top and center_y > line_bottom:
#                     line_crossed = True
#                     crossing_type = "IN"
#                 elif prev_y > line_bottom and center_y < line_top:
#                     line_crossed = True
#                     crossing_type = "OUT"

#                 if not line_crossed and prev_side:
#                     if prev_side == "above" and current_side == "below":
#                         line_crossed = True
#                         crossing_type = "IN"
#                     elif prev_side == "below" and current_side == "above":
#                         line_crossed = True
#                         crossing_type = "OUT"

#                 if line_crossed:
#                     last_counted = track_counted[track_id].get(crossing_type, 0)
#                     if last_counted == 0 or frame_number - last_counted >= CROSSING_COOLDOWN:
#                         if crossing_type == "IN":
#                             in_count += 1
#                             track_counted[track_id]["IN"] = frame_number
#                         else:
#                             out_count += 1
#                             in_count = max(0, in_count - 1)
#                             track_counted[track_id]["OUT"] = frame_number

#             track_last_side[track_id] = current_side
#             track_history_y[track_id] = center_y
#             track_last_frame[track_id] = frame_number

#     # ==========================
#     # DRAW LINE & COUNTERS
#     # ==========================
#     cv2.line(annotated_frame, (0, LINE_Y), (IMG_SIZE, LINE_Y), (0, 0, 255), 2)

#     cv2.putText(annotated_frame, f"IN: {in_count}", (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.putText(annotated_frame, f"OUT: {out_count}", (20, 80),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     writer.write(annotated_frame)
#     cv2.imshow("YOLO Video | IN-OUT People Counting", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # ==========================
# # CLEANUP
# # ==========================
# cap.release()
# writer.release()
# cv2.destroyAllWindows()
# print("✅ Video processing completed.")
# print(f"Saved output to: {OUTPUT_PATH}")







# ----- Before edits ------

# from ultralytics import YOLO
# import cv2
# import os

# # CONFIGURATION
# WEIGHTS_PATH = "best.pt"
# IMAGE_DIR = "images_for_inference"
# VIDEO_PATH = "New.mp4"
# OUTPUT_DIR = "Outputs_from_inference"

# CONF_THRES = 0.55
# IOU_THRES = 0.50
# IMG_SIZE = 640

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Load the YOLO model
# model = YOLO(WEIGHTS_PATH)

# # Image Inference (Detection only)
# print("Running inference on images...")

# for img_name in os.listdir(IMAGE_DIR):
#     img_path = os.path.join(IMAGE_DIR, img_name)

#     if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
#         continue

#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

#     results = model.predict(
#         source=img,
#         conf=CONF_THRES,
#         iou=IOU_THRES,
#         verbose=False
#     )

#     annotated_frame = results[0].plot()

#     out_path = os.path.join(OUTPUT_DIR, f"img_{img_name}")
#     cv2.imwrite(out_path, annotated_frame)

# print("Image inference completed.")



# # VIDEO INFERENCE
# print("Running inference on video...")

# cap = cv2.VideoCapture(VIDEO_PATH)


# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = IMG_SIZE
# height = IMG_SIZE

# out_video_path = os.path.join(OUTPUT_DIR, "video_inference_counted.mp4")
# writer = cv2.VideoWriter(
#     out_video_path,
#     cv2.VideoWriter_fourcc(*"mp4v"),
#     fps,
#     (width, height)
# )

# # Line position for counting
# LINE_Y = IMG_SIZE // 2

# # Tracking state
# track_history = {}
# counted_ids = set()

# in_count = 0
# out_count = 0

# while cap.isOpened():
#     ret,frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))


#     # YOLO Tracking
#     results = model.track(
#         source=frame,
#         conf=CONF_THRES,
#         iou=IOU_THRES,
#         persist=True,
#         tracker="bytetrack.yaml",
#         verbose=False
#     )

#     annotated = frame.copy()

#     if results[0].boxes.id is not None:
#         boxes = results[0].boxes.xyxy.cpu().numpy()
#         track_ids = results[0].boxes.id.cpu().numpy().astype(int)
#         classes = results[0].boxes.cls.cpu().numpy()

#         for box, track_id, cls in zip(boxes, track_ids, classes):
#             #Only people (adjust class ID if custom model)
#             if int(cls) != 0:
#                 continue

#             x1, y1, x2, y2 = box
#             center_y = int((y1 + y2) / 2)

#             #Draw bbox + ID
#             cv2.rectangle(annotated,
#                           (int(x1), int(y1)),
#                           (int(x2), int(y2)),
#                           (0, 255, 0), 2)
            
#             cv2.putText(annotated,
#                         f"ID {track_id}",
#                         (int(x1), int(y1) - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6, (0, 255, 0), 2)
            
#             #LIne-crossing Logic
#             if track_id in track_history:
#                 prev_y = track_history[track_id]

#                 # IN (top -> bottom)
#                 if prev_y < LINE_Y and center_y >= LINE_Y and track_id not in counted_ids:
#                     in_count +=1
#                     counted_ids.add(track_id)

#                 # OUT (bottom -> top)
#                 elif prev_y > LINE_Y and center_y <= LINE_Y and track_id not in counted_ids:
#                     out_count += 1
#                     counted_ids.add(track_id)
            
#             track_history[track_id] = center_y
    
    

#     # Draw reference Line
#     cv2.line(annotated, (0, LINE_Y), (IMG_SIZE, LINE_Y), (0, 0, 255), 2)


#     #Draw counters
#     cv2.putText(annotated, f"IN: {in_count}", (20,40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
#     cv2.putText(annotated, f"OUT: {out_count}", (20, 80),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     writer.write(annotated)

# cap.release()
# writer.release()

# print("Video inference with IN/OUT counting completed.")
# print(f"Saved to: {out_video_path}")