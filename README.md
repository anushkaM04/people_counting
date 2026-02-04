# Real-Time People Counting

---


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#key-features">Key Features</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project implements a **robust people detection, tracking, and people Incoming/Outgoing counting system** using **Ultralytics YOLOv11** and **YOLO ByteTracking**.  
It supports both **real-time webcam inference** and **offline video(saved) inference**, with accurate counting for number of people entering(In) and exiting(Out).



### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [YOLO Object Detection](https://docs.ultralytics.com/tasks/detect/)
* [ByteTracking](https://docs.ultralytics.com/modes/track/)
* [Roboflow](https://roboflow.com/)
* [OpenCV](https://opencv.org/)


## Getting Started

To get a local copy up and running follow these steps.

### Prerequisites

1. Check for python version. If python not installed [click here to install](https://www.python.org/downloads/).

   ```sh
   python --version
   ```


### Installation

_Follow the steps below:_

1. Clone the repo

   ```sh
   git clone https://git.spanidea.com/aiml_jodhpur/people_count.git
   ```

2. Make virtual environment:

   ```sh
   python -m venv '<name_of_your_virtual_environment>'
   ```

3. Activate your virtual environment

   ```sh
   .\'<name_of_your_virtual_environment>'\scripts\activate
   ```

3. Install requirements. **Ensure** you have latest pip package installed. 

   ```sh
   (venv) pip install -r requirements.txt
   ```

4. Download model weight from `weights/best_performing.pt` in your device. Then navigate in code folder to: `code/with_logic_saved_video.py`, and paste the path to your folder where model weight file is saved. **Remember** to give path for your result/output file to get saved.

   ```js
   WEIGHT_PATH = 'best_performing.pt'
   VIDEO_PATH = '<path/to/saved/testing/video>.mp4'
   OUTPUT_PATH = '<path/to/output/folder>.mp4'
   ```

5. Run the script using the below command **ensure you are working inside your venv**.
   ```js
   (venv) python with_logic_saved_video.py
   ```

6. **Option 2:** If you **do not have** a GPU in your device, then navigate to **GPU_inference_saved_video.ipynb** python notebook.
   Open it in Google Colab, run all the cells as they are one-by-one.
   Starting from cell 1:
   ```js
   !pip install ultralytics opencv-python
   ```

7. **Option 3:** If you have an **NVIDIA(cuda)** GPU in your local device you can run the real-time inference as directed till point 3, choosing your device as the name of your GPU. Navigate to `code/with_logic_realTime.py`, and add the following lines as  directed.

   ```
   # ADD `torch` library:
      from ultralytics import YOLO
      import cv2
      import time
      **import torch  # Imported to check for CUDA availability**


   # YOLO TRACKING (CUDA ENABLED)

   if torch.cuda.is_available():
      device = 0
      print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
   else:
      device = 'cpu'
      print("⚠️ CUDA NOT detected. Running on CPU.")

   # YOLO TRACKING (CUDA ENABLED)

    results = model.track(
        source=frame_resized,
        conf=CONF_THRES,
        iou=IOU_THRES,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        **device=device  # ✅ This forces the run onto the NVIDIA GPU**
    )

 <p align="right">(<a href="#readme-top">back to top</a>)</p>
   ```

## 🔧Configuration parameters
Key parameters you may tune:
```
CONF_THRES = 0.xx                                     # Detection confidence
IOU_THRES = 0.xx                                      # IOU threshold
IMG_SIZE = xxx                                        #Input resolution
CROSSING_COOLDOWN = xx                                # Frames before recount
LINE_TOLERENCE = x                                    # Line crossing tolerance
```
NOTES: 
- Lower IMG_SIZE improves FPS
- Lower CONF_THRES improves detection of fast/blurry people.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 📌 Key Features

- Person detection using **YOLOv11**

- Moving people overhead (camera angle) tracking using **YOLO ByteTrack**

| Document | Description |
| :--------------------- | :---------- |
| [Yolo_documentation](https://docs.ultralytics.com/modes/track/) | Tracking Mode Guide: The main guide for object tracking with YOLO models, including how to enable ByteTrack |
| [Github repo for .yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml) | Explains Configuration File: Specific parameters for ByteTrack (e.g., track_high_thresh, track_low_thresh, new_track_thresh, track_buffer) can be found in the default configuration file on GitHub. |
| [API Reference](https://docs.ultralytics.com/reference/trackers/byte_tracker/) | For developers using the Python API, a reference for the BYTETracker class and its methods is available in the Ultralytics API Reference. |
| [original_ByteTrack_documentation_on_github](https://github.com/FoundationVision/ByteTrack) | The original implementation and technical details of the ByteTrack algorithm are maintained in its official repository: The core documentation, implementation details, and information on how the algorithm works are available. |
| [Conceptual Overview](https://roboflow.com/model/bytetrack) | For a deeper understanding of the ByteTrack's algorithm mechanics, articles and guides provide excellent overviews. |
| [Article](https://datature.io/blog/introduction-to-bytetrack-multi-object-tracking-by-associating-every-detection-box) | An Introduction to BYTETrack: Multi-Object Tracking by Associating Every Detection Box. |
- Robust **IN / OUT people counting**
- Works for:
  - Centered movement
  - Left/right movement
  - Diagonal movement
  - moving people
- Handles detection gaps & ID switches
- Offline **video(saved) processing with saved output** available
- Real-time **FPS calculation & overlay**
- Windows OS-friendly (OpenCV + DirectShow)

---

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🧠 How the Counting Works

1. Each detected person is assigned a **unique tracking ID**
2. A **reference line** is drawn in the frame
3. Each person’s vertical position is tracked across frames
4. When a person crosses the line:
   - **Above → Below** → `IN`
   - **Below → Above** → `OUT`
5. A **cooldown mechanism** prevents double counting after the person is detected and counted once.
6. Counting is robust to:
   - Frame drops
   - Fast motion
   - Lateral movement
   - Temporary occlusions
7. Output video will be saved with:
- Bounding boxes
- Tracking IDs
- IN/OUT counts
- FPS overlay

---

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🎥 Output Visualization
The output video shows:
- Bounding boxes for each person
- Unique tracking IDs
- Reference counting line
- IN/OUT counters overlay
- Real-time FPS overlay

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🧪Tested Scenarios
- Slow walking
- Multiple people crossing simultaneously
- people crossing from edges
- Temporary occlusion
- Re-entry after exiting

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## ⚠️Known Limitations
- Extremely fast motion may still be missed if:
   Camera FPS is too slow
   Severe motion blur occurs
- Accuracy depends on:
   Camera placement
   Lighting conditions
   Video quality

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 📂 Repository Structure

```text
├── requirements.txt
├── people_counter.py                 # Main inference & counting script
│
├── code/
│ └── extract_frames.py               # For extracting frames from custom video dataset
│ └── GPU_inference_saved_video.ipynb # Google Colab notebook for T4 GPU inference
│ └── GPU_inference_saved_video.py    # Python script version of the Colab notebook
│ └── with_logic_realTime.py          # Real-time inference with counting logic
│ └── with_logic_saved_video.py       # Saved video inference with counting logic
│ └── without_logic_realTime.py       # Real-time inference (detection only)
│ └── without_logic_saved_video.py    # Saved video inference (detection only)
│
├── weights/                          # Model weights folder
│ └── best_performing.pt
│ └── best2.pt                               
│ └── best3.pt                                
└── README.md                         # Project documentation




