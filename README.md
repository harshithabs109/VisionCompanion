# VisionCompanion: Real-time Object Detection with YOLOv5 and Flask

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey?style=for-the-badge&logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-brightgreen?style=for-the-badge&logo=opencv)
![YOLOv5](https://img.shields.io/badge/YOLOv5-v7.0-purple?style=for-the-badge&logo=yolo)

## 🌟 Project Overview

VisionCompanion is a user-friendly web application designed for real-time object detection on various media types. It utilizes the powerful YOLOv5 deep learning model, integrated with a Flask web framework, to enable users to upload images or videos and visualize detected objects directly in their browser.

This project serves as an accessible demonstration of computer vision capabilities, abstracting away the complexities of model inference and environment setup for the end-user.

## ✨ Features

* **Image Object Detection:** Upload common image formats (JPG, PNG, GIF, BMP, TIFF) and receive an annotated image with detected objects, bounding boxes, and labels.
* **Video Object Detection:** Upload popular video formats (MP4, AVI, MOV, MKV, FLV, WEBM) and obtain a processed video output with objects tracked frame by frame.
* **YOLOv5 Integration:** Employs the efficient `yolov5s` model for robust and fast inference.
* **Intuitive Web Interface:** A straightforward web form facilitates file uploads and displays detection results.
* **Robust Error Handling:** Provides clear feedback for unsupported file types, processing failures, or missing dependencies.
* **Video Codec Compatibility:** Configured to work with common video codecs, addressing known issues like black output videos (e.g., requires OpenH264 for `avc1` codec on Windows).
* **Webcam Support (Planned/Optional):** Future capability to integrate live webcam streaming for real-time, in-browser object detection.

## 🚀 Getting Started

Follow these instructions to set up and run VisionCompanion on your local machine.

### Prerequisites

Ensure you have the following software installed before proceeding:

* **Python 3.8 or higher:** [Download Python](https://www.python.org/downloads/)
* **Git:** [Download Git](https://git-scm.com/downloads)
* **OpenH264 Library (for MP4 output on Windows):** This is crucial for `cv2.VideoWriter_fourcc(*'avc1')` to function correctly.
    * Download `openh264-1.8.0-win64.dll` from the [Cisco OpenH264 Releases page](https://github.com/cisco/openh264/releases). Ensure you pick version 1.8.0 for `win64`.
    * Place this `openh264-1.8.0-win64.dll` file inside your Python virtual environment's OpenCV library folder. The typical path is:
        `your_project_directory/venv/Lib/site-packages/cv2/`

### Installation

1.  **Clone the VisionCompanion repository:**

    ```bash
    git clone https://github.com/harshithabs109/VisionCompanion.git
    cd VisionCompanion
    ```

2.  **Clone the YOLOv5 repository as a submodule:**
    This project is structured to expect the official Ultralytics `yolov5` repository as a direct sub-directory.

    ```bash
    git submodule add https://github.com/ultralytics/yolov5.git yolov5
    git submodule update --init --recursive
    ```
    *If you already have a `yolov5` folder from a previous clone or download, ensure it's the Ultralytics one and its requirements are met.*

3.  **Create a Python Virtual Environment (Highly Recommended):**
    A virtual environment isolates project dependencies, preventing conflicts with other Python projects.

    ```bash
    python -m venv venv
    ```

4.  **Activate the Virtual Environment:**

    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS / Linux:**
        ```bash
        source venv/bin/activate
        ```

5.  **Install Project Dependencies:**
    This step installs Flask, OpenCV, NumPy, PyTorch, and all specific dependencies required by YOLOv5.

    ```bash
    pip install -r yolov5/requirements.txt
    pip install flask opencv-python numpy torch
    ```
    * **Note on PyTorch:** The `torch` installation above defaults to the CPU version. For significantly faster inference if you have a compatible NVIDIA GPU, refer to the [PyTorch website](https://pytorch.org/get-started/locally/) for specific CUDA-enabled installation commands.

6.  **Download YOLOv5s Pre-trained Weights:**
    The `app.py` script requires the `yolov5s.pt` model weights to be present within your `yolov5/` directory. If they are not already there (they usually download on first use of `detect.py` or `hub.load`), you can ensure their presence:

    ```bash
    # Option A: Trigger download via YOLOv5 hubconf (run from VisionCompanion root)
    python yolov5/models/hubconf.py yolov5s --weights yolov5s.pt --source "" --force

    # Option B: Manual Download (place in yolov5/ directory)
    # Download from: [https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)
    # Then move the downloaded file into your `C:\Users\91733\Desktop\vision-companion\yolov5\` folder.
    ```

### Running the Application

1.  **Ensure your virtual environment is activated.** (e.g., `.\venv\Scripts\activate` on Windows).
2.  **Run the Flask application from the `VisionCompanion` root directory:**

    ```bash
    python app.py
    ```

3.  **Access the web application:**
    Open your web browser and navigate to `http://127.0.0.1:5000/`

## 📁 Project Structure
```bash
VisionCompanion/
├── app.py                      # Main Flask application logic
├── yolov5/                     # Cloned YOLOv5 repository (submodule)
│   ├── models/
│   ├── utils/
│   └── ...
├── static/
│   └── results/                # Processed images and videos are saved here
│       └── (output files)
├── templates/
│   ├── index.html              # HTML for file upload form
│   ├── result.html             # HTML for displaying detection results
├── uploads/                    # Temporary storage for uploaded files
│   └── (uploaded files)
├── snapshots/  
├── runs/  
├── venv/                       # Python Virtual Environment
├── .gitignore                  # Specifies files/directories to ignore in Git
└── README.md                   # This fileVisionCompanion/
├── app.py                      # Main Flask application logic
├── yolov5/                     # Cloned YOLOv5 repository (submodule)
│   ├── yolov5s.pt              # YOLOv5 pre-trained weights
│   ├── models/
│   ├── utils/
│   └── ...
├── static/
│   └── results/                # Processed images and videos are saved here
│       └── (output files)
├── templates/
│   ├── index.html              # HTML for file upload form
│   ├── result.html             # HTML for displaying detection results
│   └── webcam.html             # HTML for live webcam feed (if added)
├── uploads/                    # Temporary storage for uploaded files
│   └── (uploaded files)
├── venv/                       # Python Virtual Environment
├── .gitignore                  # Specifies files/directories to ignore in Git
├── README.md                   # This file
├── detect_video.py
├── detect_webcam_toggle.py
├── detect.py 
│── yolov5s.pt                  # YOLOv5 pre-trained weights

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository, open issues for bugs or feature requests, and submit pull requests with improvements.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).