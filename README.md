````markdown
# 🚗 AI Powered Vehicle Counting & Tracking System

An IoT-based vehicle detection system that counts cars, trucks, and buses in real-time using **YOLOv8** and pushes live traffic data to **Firebase Firestore**.

_(Note: This repository contains the Python/AI backend code. The mobile application that visualizes this data resides in a separate repository.)_

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Raspberry%20Pi-blue)
![Tech](https://img.shields.io/badge/Tech-YOLOv8%20%7C%20Firebase%20%7C%20Python-orange)

## 🌟 Features

- **Real-time Detection:** Detects Cars, Trucks, Buses, and Motorcycles.
- **Robust Tracking:** Implements **ByteTrack** to prevent ID switching and double counting.
- **Cloud Integration:** Syncs live data to **Firebase Firestore** (which feeds the Mobile Dashboard).
- **Edge Optimization:** Compatible with NCNN format for high performance on Raspberry Pi.

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Computer Vision:** OpenCV, Ultralytics YOLOv8
- **Database:** Google Firebase (Firestore)

---

## 🚀 Installation & Setup

### 1. Windows Setup (Local Testing)

If you are running this project on your local Windows machine:

1.  Clone the repository:

    ```powershell
    git clone [https://github.com/YunusKok/AIPoweredCarTracking.git](https://github.com/YunusKok/AIPoweredCarTracking.git)
    cd AIPoweredCarTracking
    ```

2.  Create a virtual environment:

    ```powershell
    python -m venv venv
    ```

3.  **Activate the environment** and install dependencies:

    ```powershell
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

    _(Note: If you get a permission error in PowerShell, try running as Administrator)_

4.  **Firebase Setup:**

    - Place your `serviceAccountKey.json` file in the root directory.

5.  Run the detection script:
    ```powershell
    python main_car.py
    ```

### 2. Raspberry Pi Setup (Linux)

If deploying to a Raspberry Pi:

1.  Setup environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Optimize Model:** Export YOLO to NCNN format for better FPS:

    ```bash
    yolo export model=yolov8n.pt format=ncnn
    ```

3.  Run the script:
    ```bash
    python main_car.py
    ```

---

## 📂 Project Structure

```text
AIPoweredCarTracking/
├── main_car.py             # Main Python script for object detection
├── serviceAccountKey.json  # Firebase Admin SDK Key (DO NOT COMMIT THIS)
├── requirements.txt        # Python dependencies
└── RASPBERRY_PI_SETUP.txt  # Detailed setup guide for Pi
```
````

## ⚠️ Important Notes

- **Secrets:** Never push `serviceAccountKey.json` to GitHub. It is included in `.gitignore`.
- **Model:** The project uses `yolov8n.pt` by default. For Raspberry Pi, ensure you use the NCNN exported format.

## 📧 Contact

Project Link: [https://github.com/YunusKok/AIPoweredCarTracking](https://github.com/YunusKok/AIPoweredCarTracking)
