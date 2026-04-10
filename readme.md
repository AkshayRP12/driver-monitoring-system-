# AI-Based Driver Monitoring System
### Real-Time Drowsiness & Distraction Detection

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-YOLOv8-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

---

## Overview

An intelligent, real-time driver monitoring system that uses deep learning and computer vision to detect **drowsiness** and **distraction**, helping prevent road accidents caused by inattentive driving.

The system simultaneously runs two AI models on every webcam frame:
- A **CNN (TensorFlow)** that classifies eye state (open/closed) to detect drowsiness
- **YOLOv8 (PyTorch)** to detect phone usage as a distraction indicator

Both outputs feed into a **decision logic engine** that generates contextual alerts in real time.

---

## Technologies Used

| Category | Tool |
|---|---|
| Programming Language | Python 3.8+ |
| Computer Vision | OpenCV |
| Drowsiness Detection | TensorFlow / Keras (CNN) |
| Distraction Detection | PyTorch + Ultralytics YOLOv8 |
| Data Handling | NumPy |
| Face Detection | Haar Cascade Classifier |

---

## Datasets

| Dataset | Purpose | Link |
|---|---|---|
| Open/Closed Eyes Dataset | Train the CNN model to classify eye state | [Kaggle](https://www.kaggle.com/datasets/sehriyarmemmedli/open-closed-eyes-dataset) |
| COCO Dataset (via YOLOv8 pretrained) | Detect cell phones in frame | [cocodataset.org](https://cocodataset.org/) |

---

## Project Structure

```
driver_monitoring/
│
├── models/
│   ├── drowsiness_model.h5       # Trained CNN model
│   └── yolov8n.pt                # Pretrained YOLOv8 weights
│
├── dataset/
│   ├── train/
│   │   ├── open/                 # Open eye images
│   │   └── closed/               # Closed eye images
│   └── val/
│       ├── open/
│       └── closed/
│
├── src/
│   ├── camera.py                 # Webcam capture
│   ├── face_detection.py         # Haar Cascade face detection
│   ├── yolo.py                   # YOLOv8 phone detection
│   ├── drowsiness.py             # CNN eye state prediction
│   ├── decision_logic.py         # Alert generation logic
│   └── main.py                   # Entry point — integrates all modules
│
├── train_drowsiness.py           # CNN training script
└── README.md
```

---

## System Workflow

```
Webcam -> OpenCV -> Frame Processing
                        |
           +------------+------------+
           |                         |
  CNN (TensorFlow)           YOLO (PyTorch)
   Eye State Detection       Phone Detection
   (Open / Closed)           (Cell Phone)
           |                         |
           +------------+------------+
                        |
                 Decision Logic
                        |
          +-------------+-------------+
          |             |             |
   Drowsiness     Distraction    High Risk
     Alert           Alert         Alert
```

---

## Alert Logic

| Condition | Alert |
|---|---|
| Eyes closed detected | Drowsiness Alert |
| Phone detected in frame | Distraction Alert |
| Both conditions true | High Risk Alert |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/driver-monitoring-system.git
cd driver-monitoring-system
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install tensorflow torch torchvision opencv-python ultralytics numpy
```

### 4. Add Model Weights

- Place your trained CNN model at `models/drowsiness_model.h5`
- YOLOv8 nano weights (`yolov8n.pt`) will auto-download on first run, or place manually in `models/`

---

## Training the CNN Model

If you want to train the drowsiness model from scratch:

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/sehriyarmemmedli/open-closed-eyes-dataset) and organize it into the following structure:

```
dataset/
├── train/
│   ├── open/
│   └── closed/
└── val/
    ├── open/
    └── closed/
```

2. Run the training script:

```bash
python train_drowsiness.py
```

The trained model will be saved to `models/drowsiness_model.h5`.

---

## Running the System

```bash
python src/main.py
```

- Ensure your webcam is connected and accessible
- Press **`Q`** to quit the application

---

## Module Breakdown

| File | Description |
|---|---|
| `camera.py` | Initializes and reads frames from the webcam |
| `face_detection.py` | Detects face region using Haar Cascade and draws bounding boxes |
| `yolo.py` | Loads YOLOv8 and detects cell phones in each frame |
| `drowsiness.py` | Preprocesses eye region and runs CNN inference |
| `decision_logic.py` | Combines model outputs and triggers appropriate alerts |
| `main.py` | Orchestrates all modules in a unified real-time loop |

---

## Key Concepts Applied

- **Convolutional Neural Networks (CNN)** — Binary image classification (open vs. closed eyes)
- **Object Detection (YOLOv8)** — Real-time phone detection using a pretrained COCO model
- **Computer Vision** — Live video capture and frame-by-frame processing with OpenCV
- **Image Processing** — Face and eye region extraction, resizing, normalization
- **Decision Fusion** — Combining outputs from two independent models for a unified alert system

---

## Requirements

```
tensorflow>=2.10
torch>=2.0
torchvision
opencv-python
ultralytics
numpy
```

---

## Future Improvements

- [ ] Add audio/sound alerts alongside visual overlays
- [ ] Integrate head pose estimation for yawning detection
- [ ] Deploy on edge devices (Raspberry Pi / Jetson Nano)
- [ ] Add logging and alert history dashboard
- [ ] Support night-vision / IR camera input

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [Open/Closed Eyes Dataset — Kaggle](https://www.kaggle.com/datasets/sehriyarmemmedli/open-closed-eyes-dataset)
- [COCO Dataset](https://cocodataset.org/)