# 🚆 WNIR — Automatic Wagon Number and Container ID Recognition System

## Overview

WNIR is a deep learning-based railway image analysis system designed to detect wagon number regions and container identification regions from freight train images.

The system combines YOLOv8 object detection, OpenCV image preprocessing, and EasyOCR-based text recognition to automate the identification process.

## Features

- Wagon number region detection using YOLOv8.
- Container code region detection using YOLOv8.
- Image preprocessing using OpenCV.
- Alphanumeric text recognition using EasyOCR.
- Bounding-box visualization.
- Detection and OCR confidence display.
- Interactive Streamlit application.
- CSV export of recognition results.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Object Detection | YOLOv8 |
| OCR | EasyOCR |
| Image Processing | OpenCV |
| Deep Learning | PyTorch |
| Web Interface | Streamlit |
| Data Processing | NumPy, Pandas |

## Project Structure

```text
WNIR/
├── models/
│   ├── wagon_detector/
│   │   └── best.pt
│   └── container_detector/
│       └── best.pt
├── src/
│   ├── __init__.py
│   ├── detector.py
│   ├── ocr.py
│   └── pipeline.py
├── outputs/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd WNIR
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add trained model weights

Place the downloaded YOLOv8 model weights in:

```text
models/wagon_detector/best.pt
models/container_detector/best.pt
```

### 5. Run the application

```bash
streamlit run app.py
```

## System Workflow

1. Upload a freight train image.
2. Detect wagon number and container code regions.
3. Extract detected regions using bounding boxes.
4. Preprocess the cropped images.
5. Recognize alphanumeric text using EasyOCR.
6. Display annotated images and recognition results.
7. Export the results as CSV.

## Model Evaluation

The object detection models were evaluated separately.

| Model | mAP@50 | mAP@50–95 |
|---|---:|---:|
| Wagon Number Detector | 99.50% | 92.31% |
| Container Code Detector | 94.33% | 72.44% |

These are dataset-specific object detection evaluation results, not OCR accuracy or overall system accuracy.

OCR performance is evaluated separately using character error rate and exact-match accuracy. Recognition can be affected by image quality, text orientation, lighting, and character visibility.

## Limitations

- OCR recognition may be inaccurate for blurred or low-resolution text.
- Similar-looking characters can be confused.
- Detection and recognition performance depend on image quality.
- The system is currently an image-based prototype rather than a real-time railway monitoring deployment.

## Future Scope

- Improve OCR accuracy using a larger verified dataset.
- Integrate a dedicated trained text-recognition model.
- Extend the system to railway video streams.
- Add object tracking and duplicate identification removal.
- Develop a railway identification database.
- Explore real-time deployment.


