# MPKIT-CC: Enhanced MediaPipe Toolkit 🚀

[![PyPI Version](https://img.shields.io/pypi/v/mpkit-cc?color=blue&logo=pypi)](https://pypi.org/project/mpkit-cc/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue?logo=python)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-FF6F00?logo=mediapipe&logoColor=white)](https://mediapipe.dev/)

**MPKIT-CC** is a powerful Python wrapper that simplifies MediaPipe's computer vision capabilities with real-time FPS monitoring, customizable visualizations, and intuitive APIs for hand tracking, face detection, facial landmarks, and pose estimation.

![Feature Showcase](https://via.placeholder.com/800x450/2d2d2d/ffffff?text=Hand+Tracking+%7C+Face+Mesh+%7C+Pose+Estimation+%7C+Real-time+FPS)

## ✨ Key Features

- **Real-time FPS Monitoring** - Built-in frame rate display with customizable styling
- **Multi-Feature Detection**:
  - ✋ Hand tracking with 21 landmarks per hand
  - 😊 Face detection with bounding boxes
  - 👁️ 468-point facial mesh detection
  - 🧍 Full-body pose estimation (33 landmarks)
- **Custom Visual Styles**:
  - 🎨 Predefined color constants (RED, GREEN, BLUE, etc.)
  - 🖌️ Choice between default MediaPipe styles or custom drawings
  - 🔗 Toggleable landmark connections
- **Flexible Configuration**:
  - 📷 Camera settings adjustment (resolution, FPS)
  - ⚙️ Detection confidence thresholds
  - 🖼️ Multiple image format support (BGR, RGB, grayscale)

## 📦 Installation

```bash
pip install mpkit-cc
```

## 🧪 Example Test Code

Here's a complete demonstration showcasing all features of MPKIT-CC with real-time FPS monitoring:

```python
from mpkit_cc import Mptools
from time import time
from cv2 import imshow, waitKey, destroyAllWindows

# Initialize with custom settings
obj = Mptools(
    image_mode=False,    # Video stream mode
    cam_index=0,         # Default camera
    win_width=640,       # Frame width
    win_height=360,      # Frame height
    cam_fps=30,          # Target FPS
    hand_no=2,           # Detect up to 2 hands
    face_no=1,           # Detect up to 1 face
    tol1=0.5,            # Detection confidence
    tol2=0.5             # Tracking confidence
)

# Start camera
cam = obj.init()
start_time = time()

while cam.isOpened():
    success, image = cam.read()
    if not success:
        print("Ignoring empty frame...")
        continue
    
    # Uncomment the detectors you want to use:
    
    # Hand detection with connections (MediaPipe default style)
    hand_data = obj.find_Hands(
        image=image,
        mode="BGR",
        hand_connection=True,
        show_detect=True,
        detection_style=1
    )
    
    # Face detection with bounding box
    face_data = obj.find_face(
        image=image,
        mode="BGR",
        show_detect=True,
        boundary=True
    )
    
    # Face mesh with 3D connections
    face_meshs = obj.find_face_mesh(
        image=image,
        mode="BGR",
        face_connection=True,
        face_connection_3d=True,
        show_detect=True
    )
    
    # Pose estimation with custom styling
    poses = obj.find_pose(
        image=image,
        mode="BGR",
        body_connection=True,
        show_detect=True,
        detection_style=0
    )
    
    # Print results if detections found
    if hand_data and hand_data != ([], [], []):
        print(f"Hands detected: {hand_data[1]} (Confidence: {hand_data[2]}%)")
    
    if face_data and face_data != ([], [], []):
        print(f"Face detected (Confidence: {face_data[2]}%)")
    
    if face_meshs:
        print(f"Face mesh points: {len(face_meshs)} landmarks")
    
    if poses:
        print(f"Body pose points: {len(poses)} landmarks")
    
    # Calculate and display FPS
    end_time = time()
    fps = int(1 / (end_time - start_time))
    start_time = end_time
    image = obj.show_FPS(
        image=image,
        mode="BGR",
        fps_rate=fps,
        fore_bg=Mptools.YELLOW,
        back_bg=Mptools.RED
    )
    
    # Display output
    imshow("MPKIT-CC Real-time Detection", image)
    
    # Exit on 'q' key press
    if waitKey(1) == ord("q"):
        break

# Cleanup
cam.release()
destroyAllWindows()
```
