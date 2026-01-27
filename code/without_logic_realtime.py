# For running realtime inference using webcam and locally saved weights without the in-out people counting logic




# from ultralytics import YOLO
# import cv2


# # CONFIGURATIONS
# WEIGHTS_PATH = r"C:\Users\Spanidea-LT06\Downloads\myPOC\best.pt"
# CONF_THRES = 0.55
# IOU_THRES = 0.50
# CAMERA_ID = 0


# # LOAD MODEL

# model = YOLO(WEIGHTS_PATH)

# # OPEN CAMERA

# cap = cv2.VideoCapture(CAMERA_ID)

# if not cap.isOpened():
#     raise RuntimeError("Could not open Webcam!")

# print("Webcam opened. Press 'q' to quit.")


# # REAL-TIME INFERENCE LOOP

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame.")
#         break

#     #RUN INFERENCE (stream=False because we handle frames manually)
#     results = model(
#         frame,
#         conf=CONF_THRES,
#         iou=IOU_THRES,
#         stream=False,
#         verbose=True
#     )

#     #DRAW DETECTIONS ON FRAME
#     annotated_frame = results[0].plot()

#     #DISPLAY FRAME
#     cv2.imshow("YOLO Webcam Inference", annotated_frame)

#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

#     # CLAN-UP
#     cap.release()
#     cv2.destroyAllWindows()
#     print("Webcam inference stopped.")
    


# from ultralytics import YOLO
# import cv2
# import time

# # ==========================
# # CONFIG
# # ==========================
# WEIGHTS_PATH = "C:\\Users\\Spanidea-LT06\\Downloads\\myPOC\\best.pt"
# CONF_THRES = 0.25
# IOU_THRES = 0.45
# CAMERA_ID = 0
# #IMG_SIZE = 640  # Standard YOLO input size

# # ==========================
# # LOAD MODEL
# # ==========================
# model = YOLO(WEIGHTS_PATH)

# # Uncomment if you have NVIDIA GPU
# # model.to("cuda")

# # ==========================
# # OPEN CAMERA (WINDOWS SAFE)
# # ==========================
# cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

# if not cap.isOpened():
#     raise RuntimeError("❌ Could not open webcam")

# # Lower capture resolution → faster read
# # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# print("✅ Webcam opened. Press 'q' to quit.")
# time.sleep(1.0)  # camera warm-up

# # ==========================
# # REAL-TIME INFERENCE LOOP
# # ==========================
# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("⚠️ Failed to grab frame, retrying...")
#         time.sleep(0.1)
#         continue

#     # 🔹 Resize to YOLO input size
#     #frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

#     # Inference
#     results = model(
#         conf=CONF_THRES,
#         iou=IOU_THRES,
#         verbose=False
#     )

#     # Draw predictions
#     annotated_frame = results[0].plot()

#     # Display
#     cv2.imshow("YOLO Webcam Inference", annotated_frame)

#     # Quit
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # ==========================
# # CLEANUP
# # ==========================
# cap.release()
# cv2.destroyAllWindows()
# time.sleep(0.5)
# print("🛑 Webcam released cleanly.")











from ultralytics import YOLO
import cv2
import time

# ==========================
# CONFIG
# ==========================
WEIGHTS_PATH = "C:\\Users\\Spanidea-LT06\\Downloads\\myPOC\\weights\\best.pt"
CONF_THRES = 0.25
IOU_THRES = 0.45
CAMERA_ID = 0
IMG_SIZE = 640

# ==========================
# LOAD MODEL
# ==========================
model = YOLO(WEIGHTS_PATH)


# ==========================
# OPEN WEBCAM (WINDOWS SAFE)
# ==========================
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✅ Webcam opened. Press 'q' to quit.")
time.sleep(1.0)

# ==========================
# REAL-TIME LOOP
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        time.sleep(0.1)
        continue

    # Resize frame for YOLO
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # ✅ IMPORTANT: pass frame as source
    results = model.predict(
        source=frame_resized,
        conf=CONF_THRES,
        iou=IOU_THRES,
        verbose=False,
        stream=False
    )

    # Draw detections
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLO Webcam Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================
# CLEANUP
# ==========================
cap.release()
cv2.destroyAllWindows()
time.sleep(0.5)
print("🛑 Webcam released cleanly.")



# import cv2

# def find_camera_index():
#     # Check the first 5 possible indices
#     # for index in range(5):
#     cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("⚠️ Failed to grab frame")
#             # time.sleep(0.1)
#             continue
#         cv2.imshow("YOLO Webcam Inference", frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     # time.sleep(0.5)
#     print("🛑 Webcam released cleanly.")


#     # Resize frame for YOLO
#     # frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))


#         # if cap.isOpened():
#         #     print(f"✅ Camera found at index: {index}")
#         #     cap.release()
#         # else:
#         #     print(f"❌ No camera at index: {index}")

# find_camera_index()
