# CodeAlpha_ObjectDetection

🔍 **Real-Time Object Detection & Tracking**  
**Internship Task 4 — CodeAlpha**

---

## 📌 About This Project

This project implements **real-time object detection and tracking** using:

- **YOLOv8** (via ultralytics) for object detection  
- **Deep SORT** for tracking objects across frames  
- **OpenCV** for video input/output (webcam or video files)  

It fulfills all internship requirements:

✔ Real-time video input (webcam or video file)  
✔ Pre-trained model for object detection  
✔ Bounding boxes with tracking IDs  
✔ Python implementation with clear, real-time results  

---

## 🚀 Features

- Detects objects in every video frame  
- Tracks detected objects with unique IDs  
- Works with multiple video files or webcam input  
- Outputs video frames with bounding boxes and labels  

---

 📦 Installation

1. **Clone the repository**

git clone https://github.com/Urooj25/CodeAlpha_ObjectDetection.git
cd CodeAlpha_ObjectDetection

2- Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

3-Install dependencies
pip install -r requirements.txt


**Usage**
Run on Video Files
Place your video(s) in the videos/ folder or detect real time object
Run the main script:
python yolo_object_tracking.py
Use Webcam


**Code Overview**

yolo_object_tracking.py — main detection & tracking script
videos/ — sample video(s) for testing
models/ — YOLO pre-trained weights 
requirements.txt — all Python dependencies



**Future Improvements**

GUI / Dashboard — Let users select videos or webcam dynamically
Save Output Videos — Add cv2.VideoWriter to save processed videos
Custom Object Classes — Fine-tune YOLO to detect specific objects more accurately
Performance Optimization — Use GPU, batch frames, or smaller models for real-time speed
Analytics / Statistics — Count objects, track motion paths, generate logs for analysis
Mobile Deployment — Convert model to ONNX / TensorRT for edge/mobile devices


object-detection • tracking • YOLOv8 • DeepSORT • OpenCV • Python


---



























