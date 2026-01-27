##########---------Previous code Trail-3 ---------########## 

from ultralytics import YOLO
import cv2
import time

# ==========================
# CONFIG
# ==========================
WEIGHTS_PATH = "C:\\Users\\Spanidea-LT06\\Downloads\\myPOC\\weights\\best.pt"
CONF_THRES = 0.55
IOU_THRES = 0.50
CAMERA_ID = 0
IMG_SIZE = 512

# Reference line (0-position)
LINE_Y = IMG_SIZE // 2

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
# TRACKING STATE (UPDATED)
# ==========================
track_last_side = {}   # track_id -> "above" or "below"
track_history_y = {}   # track_id -> previous center_y (for delta calculation)
track_first_seen = {}  # track_id -> frame number when first seen
track_last_frame = {}  # track_id -> last frame number seen
track_counted = {}     # track_id -> dict: {"IN": last_frame_counted, "OUT": last_frame_counted}
MAX_TRACK_GAP = 15     # Maximum frames to keep state for disappeared tracks (increased)
CROSSING_COOLDOWN = 25  # Frames before allowing same direction crossing again (prevents oscillation)
LINE_TOLERANCE = 5      # Tolerance zone around LINE_Y for crossing detection

in_count = 0
out_count = 0
frame_number = 0

# ==========================
# REAL-TIME LOOP
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        time.sleep(0.1)
        continue

    frame_number += 1
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
    current_frame_track_ids = set()
    previous_track_ids = set(track_last_side.keys())
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy()
        
        current_frame_track_ids = set(track_ids.tolist())
        disappeared_ids = previous_track_ids - current_frame_track_ids
        new_ids = current_frame_track_ids - previous_track_ids
        
        # FIX 4: Try to match new track IDs with disappeared ones based on position (handles track ID reassignment)
        # This helps when moving laterally causes track ID changes
        if disappeared_ids and new_ids:
            # Store positions of disappeared tracks
            disappeared_positions = {}
            for tid in disappeared_ids:
                if tid in track_history_y:
                    disappeared_positions[tid] = {
                        "y": track_history_y[tid],
                        "side": track_last_side.get(tid),
                        "counted": track_counted.get(tid, {"IN": 0, "OUT": 0}).copy()
                    }
            
            # Get positions of new tracks (only people, class 0)
            new_track_positions = {}
            for idx, (new_tid, new_box, new_cls) in enumerate(zip(track_ids, boxes, classes)):
                if new_tid in new_ids and int(new_cls) == 0:  # Only match people
                    new_center_y = int((new_box[1] + new_box[3]) / 2)
                    new_track_positions[new_tid] = new_center_y
            
            # Try to match new tracks with disappeared ones
            for new_tid in list(new_ids):
                if new_tid not in new_track_positions:
                    continue
                new_center_y = new_track_positions[new_tid]
                
                # Find closest disappeared track
                best_match = None
                best_distance = float('inf')
                for old_tid, old_pos in disappeared_positions.items():
                    distance = abs(new_center_y - old_pos["y"])
                    # IMPROVED: More lenient matching for lateral movement (increased threshold to 150 pixels)
                    # Also consider horizontal distance if we track center_x
                    if distance < 150 and distance < best_distance:
                        # Check if same side or very close (more lenient for lateral movement)
                        new_side = "above" if new_center_y < LINE_Y else "below"
                        # Match if same side, or if within 80 pixels (more lenient)
                        if old_pos["side"] == new_side or distance < 80:
                            best_match = old_tid
                            best_distance = distance
                
                # If match found, transfer state
                if best_match is not None:
                    # Transfer state
                    track_last_side[new_tid] = disappeared_positions[best_match]["side"]
                    track_history_y[new_tid] = disappeared_positions[best_match]["y"]
                    # Transfer counted state but reset if cooldown has passed
                    old_counted = disappeared_positions[best_match]["counted"]
                    new_counted = {"IN": 0, "OUT": 0}
                    # Only keep counted state if recent (within cooldown)
                    for direction in ["IN", "OUT"]:
                        if old_counted.get(direction, 0) > 0:
                            frames_since = frame_number - old_counted[direction]
                            if frames_since < CROSSING_COOLDOWN:
                                new_counted[direction] = old_counted[direction]
                    track_counted[new_tid] = new_counted
                    track_first_seen[new_tid] = track_first_seen.get(best_match, frame_number)
                    track_last_frame[new_tid] = frame_number
                    # Remove from disappeared list
                    disappeared_ids.discard(best_match)
                    new_ids.discard(new_tid)

        for box, track_id, cls in zip(boxes, track_ids, classes):

            # Only count people (class 0 for COCO; adjust if custom)
            if int(cls) != 0:
                continue

            x1, y1, x2, y2 = box
            center_y = int((y1 + y2) / 2)
            center_x = int((x1 + x2) / 2)
            
            # Calculate delta_y for speed detection
            delta_y = 0
            if track_id in track_history_y:
                delta_y = center_y - track_history_y[track_id]
            if track_id not in track_first_seen:
                track_first_seen[track_id] = frame_number

            # Draw bounding box
            cv2.rectangle(
                annotated_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0), 2
            )

            # Draw ID
            cv2.putText(
                annotated_frame,
                f"ID {track_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

            # ==========================
            # IMPROVED IN / OUT LOGIC (HANDLES FAST MOVEMENT & LATERAL TRACKING)
            # ==========================
            current_side = "above" if center_y < LINE_Y else "below"

            # Initialize counted dict for this track if needed
            if track_id not in track_counted:
                track_counted[track_id] = {"IN": 0, "OUT": 0}

            # Check if we have previous state
            if track_id in track_history_y:
                prev_y = track_history_y[track_id]
                prev_side = track_last_side.get(track_id)
                
                # IMPROVED FIX 1: Detect crossing with tolerance zone and handle edge cases
                line_crossed = False
                crossing_type = None
                
                # Use tolerance zone for more robust detection (handles lateral movement)
                line_top = LINE_Y - LINE_TOLERANCE
                line_bottom = LINE_Y + LINE_TOLERANCE
                
                # Check if line was crossed between previous and current position
                # Above to below = IN
                if prev_y < line_top and center_y > line_bottom:
                    line_crossed = True
                    crossing_type = "IN"
                # Below to above = OUT
                elif prev_y > line_bottom and center_y < line_top:
                    line_crossed = True
                    crossing_type = "OUT"
                # Handle cases where person crosses exactly at the line
                elif (prev_y <= LINE_Y and center_y > LINE_Y) or (prev_y < line_top and center_y >= line_top):
                    line_crossed = True
                    crossing_type = "IN"
                elif (prev_y >= LINE_Y and center_y < LINE_Y) or (prev_y > line_bottom and center_y <= line_bottom):
                    line_crossed = True
                    crossing_type = "OUT"
                
                # IMPROVED FIX 2: Also check side-based transition (for normal speed and edge cases)
                if not line_crossed and prev_side:
                    if prev_side == "above" and current_side == "below":
                        # Additional check: ensure we're not just oscillating
                        if abs(center_y - LINE_Y) > LINE_TOLERANCE or abs(prev_y - LINE_Y) > LINE_TOLERANCE:
                            line_crossed = True
                            crossing_type = "IN"
                    elif prev_side == "below" and current_side == "above":
                        if abs(center_y - LINE_Y) > LINE_TOLERANCE or abs(prev_y - LINE_Y) > LINE_TOLERANCE:
                            line_crossed = True
                            crossing_type = "OUT"
                
                # IMPROVED FIX 3: Allow multiple crossings with cooldown (prevents oscillation, allows legitimate re-crossings)
                if line_crossed:
                    last_counted_frame = track_counted[track_id].get(crossing_type, 0)
                    frames_since_last = frame_number - last_counted_frame
                    
                    # Only count if enough frames have passed (cooldown) or if crossing opposite direction
                    if frames_since_last >= CROSSING_COOLDOWN or last_counted_frame == 0:
                        if crossing_type == "IN":
                            in_count += 1
                            track_counted[track_id]["IN"] = frame_number
                        elif crossing_type == "OUT":
                            out_count += 1
                            in_count = max(0, in_count - 1)
                            track_counted[track_id]["OUT"] = frame_number
            else:
                # First detection of this track - initialize state
                track_counted[track_id] = {"IN": 0, "OUT": 0}

            # Update state
            track_last_side[track_id] = current_side
            track_history_y[track_id] = center_y
            track_last_frame[track_id] = frame_number

    # Clean up disappeared tracks after processing all detections
    disappeared_after_processing = previous_track_ids - current_frame_track_ids
    if disappeared_after_processing:
        for tid in disappeared_after_processing:
            frames_since_last_seen = frame_number - track_last_frame.get(tid, frame_number)
            # FIX 3: Keep state for disappeared tracks for MAX_TRACK_GAP frames (handles detection gaps)
            if frames_since_last_seen > MAX_TRACK_GAP:
                # Remove old state if track hasn't reappeared
                if tid in track_last_side:
                    del track_last_side[tid]
                if tid in track_history_y:
                    del track_history_y[tid]
                if tid in track_counted:
                    del track_counted[tid]
    
    # ==========================
    # DRAW LINE & COUNTERS
    # ==========================
    cv2.line(
        annotated_frame,
        (0, LINE_Y),
        (IMG_SIZE, LINE_Y),
        (0, 0, 255), 2
    )

    cv2.putText(
        annotated_frame,
        f"IN: {in_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 2
    )

    cv2.putText(
        annotated_frame,
        f"OUT: {out_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 255), 2
    )

    cv2.imshow("YOLO Webcam | IN-OUT People Counting", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================
# CLEANUP
# ==========================
cap.release()
cv2.destroyAllWindows()
time.sleep(0.5)
print("🛑 Webcam released cleanly.")



































##########---------Previous code Trail-2 ---------########## 
# from ultralytics import YOLO
# import cv2
# import time

# # ==========================
# # CONFIG
# # ==========================
# WEIGHTS_PATH = "C:\\Users\\Spanidea-LT06\\Downloads\\myPOC\\weights\\best.pt"
# CONF_THRES = 0.50
# IOU_THRES = 0.45
# CAMERA_ID = 0
# IMG_SIZE = 640

# # Reference line (0-position)
# LINE_Y = IMG_SIZE // 2

# # Tolerance zone around the line (IMPORTANT)
# LINE_MARGIN = 30   # pixels (increase if needed)

# # ==========================
# # LOAD MODEL
# # ==========================
# model = YOLO(WEIGHTS_PATH)

# # ==========================
# # OPEN WEBCAM (WINDOWS SAFE)
# # ==========================
# cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

# if not cap.isOpened():
#     raise RuntimeError("❌ Could not open webcam")

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# print("✅ Webcam opened. Press 'q' to quit.")
# time.sleep(1.0)

# # ==========================
# # TRACKING STATE
# # ==========================
# track_history = {}      # track_id -> previous center_y
# counted_ids = set()     # avoid double counting

# in_count = 0
# out_count = 0

# # ==========================
# # REAL-TIME LOOP
# # ==========================
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("⚠️ Failed to grab frame")
#         time.sleep(0.1)
#         continue

#     # Resize frame for YOLO
#     frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
#     annotated_frame = frame_resized.copy()

#     # YOLO tracking (ByteTrack)
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
#     if results[0].boxes.id is not None:
#         boxes = results[0].boxes.xyxy.cpu().numpy()
#         track_ids = results[0].boxes.id.cpu().numpy().astype(int)
#         classes = results[0].boxes.cls.cpu().numpy()

#         for box, track_id, cls in zip(boxes, track_ids, classes):

#             # Only count people (class 0 for COCO, adjust if custom)
#             if int(cls) != 0:
#                 continue

#             x1, y1, x2, y2 = box
#             center_y = int((y1 + y2) / 2)

#             # Draw bounding box
#             cv2.rectangle(
#                 annotated_frame,
#                 (int(x1), int(y1)),
#                 (int(x2), int(y2)),
#                 (0, 255, 0), 2
#             )

#             # Draw ID
#             cv2.putText(
#                 annotated_frame,
#                 f"ID {track_id}",
#                 (int(x1), int(y1) - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6, (0, 255, 0), 2
#             )

#             # ==========================
#             # IMPROVED IN / OUT LOGIC
#             # ==========================
#             if track_id in track_history:
#                 prev_y = track_history[track_id]
#                 dy = center_y - prev_y  # movement direction

#                 # Check if inside crossing zone
#                 in_zone = (LINE_Y - LINE_MARGIN) <= center_y <= (LINE_Y + LINE_MARGIN)

#                 if in_zone and track_id not in counted_ids:

#                     # Moving DOWN → IN
#                     if dy > 0:
#                         in_count += 1
#                         counted_ids.add(track_id)

#                     # Moving UP → OUT
#                     elif dy < 0:
#                         out_count += 1
#                         in_count = max(0, in_count - 1)
#                         counted_ids.add(track_id)

#             track_history[track_id] = center_y

#     # ==========================
#     # DRAW ZONE & COUNTERS
#     # ==========================
#     # Draw counting zone
#     cv2.rectangle(
#         annotated_frame,
#         (0, LINE_Y - LINE_MARGIN),
#         (IMG_SIZE, LINE_Y + LINE_MARGIN),
#         (0, 0, 255), 2
#     )

#     # Draw counters
#     cv2.putText(
#         annotated_frame,
#         f"IN: {in_count}",
#         (20, 40),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1, (0, 255, 0), 2
#     )

#     cv2.putText(
#         annotated_frame,
#         f"OUT: {out_count}",
#         (20, 80),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1, (0, 0, 255), 2
#     )

#     # Show output
#     cv2.imshow("YOLO Webcam | IN-OUT People Counting", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # ==========================
# # CLEANUP
# # ==========================
# cap.release()
# cv2.destroyAllWindows()
# time.sleep(0.5)
# print("🛑 Webcam released cleanly.")

























































##########---------Previous code Trail-1 ---------##########     

# from ultralytics import YOLO
# import cv2
# import time

# # ==========================
# # CONFIG
# # ==========================
# WEIGHTS_PATH = "C:\\Users\\Spanidea-LT06\\Downloads\\myPOC\\weights\\best.pt"
# CONF_THRES = 0.25
# IOU_THRES = 0.45
# CAMERA_ID = 0
# IMG_SIZE = 640

# # Reference line (0-position)
# LINE_Y = IMG_SIZE // 2  # middle of frame

# # ==========================
# # LOAD MODEL
# # ==========================
# model = YOLO(WEIGHTS_PATH)

# # ==========================
# # OPEN WEBCAM (WINDOWS SAFE)
# # ==========================
# cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

# if not cap.isOpened():
#     raise RuntimeError("❌ Could not open webcam")

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# print("✅ Webcam opened. Press 'q' to quit.")
# time.sleep(1.0)

# # ==========================
# # TRACKING STATE
# # ==========================
# track_history = {}      # track_id -> previous center_y
# counted_ids = set()     # to avoid double counting

# in_count = 0
# out_count = 0

# # ==========================
# # REAL-TIME LOOP
# # ==========================
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("⚠️ Failed to grab frame")
#         time.sleep(0.1)
#         continue

#     # Resize frame for YOLO
#     frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
#     annotated_frame = frame_resized.copy()

#     # ✅ YOLO tracking (IMPORTANT)
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
#     if results[0].boxes.id is not None:
#         boxes = results[0].boxes.xyxy.cpu().numpy()
#         track_ids = results[0].boxes.id.cpu().numpy().astype(int)
#         classes = results[0].boxes.cls.cpu().numpy()

#         for box, track_id, cls in zip(boxes, track_ids, classes):

#             # Only count people (class 0 for COCO; adjust if custom)
#             if int(cls) != 0:
#                 continue

#             x1, y1, x2, y2 = box
#             center_y = int((y1 + y2) / 2)

#             # Draw bounding box
#             cv2.rectangle(
#                 annotated_frame,
#                 (int(x1), int(y1)),
#                 (int(x2), int(y2)),
#                 (0, 255, 0), 2
#             )

#             # Draw ID
#             cv2.putText(
#                 annotated_frame,
#                 f"ID {track_id}",
#                 (int(x1), int(y1) - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6, (0, 255, 0), 2
#             )

#             # ==========================
#             # IN / OUT LOGIC
#             # ==========================
#             if track_id in track_history:
#                 prev_y = track_history[track_id]

#                 # IN → top to bottom
#                 if prev_y < LINE_Y and center_y >= LINE_Y and track_id not in counted_ids:
#                     in_count += 1
#                     counted_ids.add(track_id)

#                 # OUT → bottom to top
#                 elif prev_y > LINE_Y and center_y <= LINE_Y and track_id not in counted_ids:
#                     out_count += 1
#                     in_count = max(0, in_count - 1)
#                     counted_ids.add(track_id)

#             track_history[track_id] = center_y

#     # ==========================
#     # DRAW LINE & COUNTERS
#     # ==========================
#     cv2.line(
#         annotated_frame,
#         (0, LINE_Y),
#         (IMG_SIZE, LINE_Y),
#         (0, 0, 255), 2
#     )

#     cv2.putText(
#         annotated_frame,
#         f"IN: {in_count}",
#         (20, 40),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1, (0, 255, 0), 2
#     )

#     cv2.putText(
#         annotated_frame,
#         f"OUT: {out_count}",
#         (20, 80),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1, (0, 0, 255), 2
#     )

#     # Show output
#     cv2.imshow("YOLO Webcam | IN-OUT People Counting", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # ==========================
# # CLEANUP
# # ==========================
# cap.release()
# cv2.destroyAllWindows()
# time.sleep(0.5)
# print("🛑 Webcam released cleanly.")




