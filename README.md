# 🎯 VISOR - Advanced Object Detection & Tracking System

A professional-grade, browser-based real-time object detection and tracking system
built with **YOLO11, YOLOv8, DeepSORT, OpenCV, and Flask**.

> **VISOR** — *See Everything. Miss Nothing.*

### ✨ Highlights
- 10 switchable YOLO models (YOLO11 + YOLOv8, nano to extra-large)
- DeepSORT multi-object tracking with persistent IDs, speed and direction
- **Start / Pause / Stop** controls — the camera never opens until you press Start
- **Video seek bar** with current time / total duration (video files)
- ROI zone alerts, movement heatmap, class filter, CSV logging, snapshots

---

## 📁 Project Structure

```
VISOR-Object-Detection-and-Tracking/
│
├── app.py                     ← Flask server, video thread, Start/Pause/Stop + seek logic
├── config.py                  ← All settings & constants (edit here!)
├── requirements.txt           ← Python dependencies
│
├── modules/
│   ├── detector.py            ← YOLO11 / YOLOv8 detection wrapper
│   ├── tracker.py             ← DeepSORT tracking wrapper
│   ├── analytics.py           ← Speed, direction, counts, CSV logging
│   ├── heatmap.py             ← Movement heatmap generator
│   └── roi_manager.py         ← ROI zone alerts
│
├── templates/
│   └── index.html             ← Web dashboard UI
│
├── static/
│   ├── css/style.css          ← Cyberpunk dark theme
│   └── js/main.js             ← Playback controls, seek bar, ROI canvas, analytics polling
│
└── logs/
    └── detections.csv         ← Auto-generated detection log
```

---

## 🛠️ Technologies Used

* Python 3.9+
* Flask
* OpenCV
* Ultralytics YOLO11
* Ultralytics YOLOv8
* DeepSORT
* NumPy
* HTML5
* CSS3
* JavaScript


## ⚡ Quick Setup (Step by Step)

### Step 1 — Install Python 3.9+
Download from: https://python.org/downloads  
✅ Check **"Add Python to PATH"** during installation

### Step 2 — Install dependencies
Open a terminal in the project folder and run:
```bash
pip install -r requirements.txt
```
> This installs Flask, Ultralytics (YOLO11 / YOLOv8), DeepSORT, OpenCV, and NumPy.
> `yolo11n.pt` and `yolov8n.pt` are included. Other model weights (~5–130MB) download automatically the first time you select them.

### Step 3 — Run the server
```bash
python app.py
```

### Step 4 — Open the dashboard
Open your browser and go to:
```
http://localhost:5000
```

### Step 5 — Press ▶ Start
The dashboard opens on a standby screen and the camera stays **closed**.
Click **▶ Start** in the header to begin detection.

---

## 🤖 YOLO Models Included & Supported

This project is designed to work with multiple Ultralytics YOLO models for object detection.

### Included in Repository

The following lightweight models are included for immediate testing:

| Model                  | Size    | Purpose                           |
| ---------------------- | ------- | --------------------------------- |
| YOLO11n (`yolo11n.pt`) | ~5.6 MB | Lightweight, fast inference       |
| YOLOv8n (`yolov8n.pt`) | ~6.5 MB | Lightweight, balanced performance |

### Additional Models Supported

The system was developed and tested with the following models:

| Model                  | Size    |
| ---------------------- | ------- |
| YOLO11m (`yolo11m.pt`) | ~40 MB  |
| YOLO11l (`yolo11l.pt`) | ~51 MB  |
| YOLO11x (`yolo11x.pt`) | ~115 MB |
| YOLOv8l (`yolov8l.pt`) | ~88 MB  |
| YOLOv8x (`yolov8x.pt`) | ~137 MB |

Large model weight files are not included in this repository to keep the project lightweight and within GitHub file size limits.

Users may download additional model weights directly from Ultralytics and place them in the project root directory.


## 🎮 Features & How to Use

### ▶ Start / ⏸ Pause / ⏹ Stop
The header has three playback buttons and a live state badge.

| State | Camera | Detection | What you see |
|-------|--------|-----------|--------------|
| **STOPPED** | Closed | Off | VISOR standby screen |
| **RUNNING** | Open | Active | Live annotated video |
| **PAUSED** | Open | Frozen | Last frame with a PAUSED banner |

- Server starts in **STOPPED**, so the webcam never turns on by accident
- **Pause** freezes the pipeline instantly; **Resume** continues from the same frame
- **Stop** releases the camera and returns to standby

### ⏩ Video Seek Bar (video files only)
- Upload a video, press Start, and a seek bar appears below the video
- Shows **current time** and **total duration** (e.g. `03:12 / 08:45`)
- Drag the slider to jump to any position, for example to start analysing from minute 3
- Tracker and analytics reset after a seek so IDs stay clean
- Hidden automatically when the source is a webcam

### 🎯 DeepSORT Tracking
- Each object gets a **unique ID** that persists across frames
- IDs survive brief occlusions (object hidden behind something)
- Based on **Kalman filter** (position prediction) + **appearance matching**
- Toggle on/off with the switch in Controls tab

### 📊 Analytics Dashboard
- Live FPS counter
- Current objects in frame
- Total unique IDs ever tracked
- Per-class object counts
- All data updates every 700ms automatically

### 🏃 Speed & Direction
- Speed shown in **px/s** next to each object label
- Direction shown as compass arrow: N / NE / E / SE / S / SW / W / NW
- A coloured arrow on the object shows movement direction
- Motion trail shows the past path

### 📦 ROI Zone Alerts
1. Go to the **ROI Zones** tab in the sidebar
2. Click **Draw Zone** and enter a name
3. Click and drag on the video to draw a rectangle
4. The zone shows how many objects are inside it in real time
5. Zone crossings are saved to the CSV log
- Up to 5 zones supported
- Each zone has a distinct colour

### 🌊 Movement Heatmap
- Shows where objects spend the most time
- Blue = rare, Green = moderate, Red = frequent
- Live preview in the **Heatmap** tab
- Toggle "blend onto video" to overlay on the main feed
- Temporal decay: old visits fade gradually
- Reset anytime with the Clear button

### 💾 CSV Log Download
Every detection is logged with:
- Timestamp, frame number, track ID, class, confidence
- Bounding box coordinates, centroid position
- Speed (px/s), direction, ROI zone name

Click **Download CSV** to get the full log file.

### 🔎 Class Filter
- Filter to detect only specific object types
- Example: check only "person" and "car" to ignore everything else
- Supports all 80 COCO classes

### 📷 Video Source
- **Webcam**: Select index 0 (default) or 1 (second camera)
- **Video File**: Upload an .mp4 / .avi file (saved to `logs/`), then press Start
- Switch sources without restarting the server

---

## ⚙️ Configuration (config.py)

Edit `config.py` to customize behaviour:

```python
DEFAULT_CONFIDENCE = 0.40    # Lower = more detections (less precise)
TRAIL_LENGTH       = 30      # Longer trail = more history shown
HEATMAP_DECAY      = 0.995   # Closer to 1.0 = slower fade
SPEED_SCALE        = 0.05    # Calibrate for your scene (px/frame → km/h)
TARGET_FPS         = 30      # Processing frame rate cap
```

---

## 🧠 How It All Works

```
Webcam/Video  (starts only after ▶ Start)
    ↓
OpenCV (frame capture, seek support for files)
    ↓
YOLO11 / YOLOv8 (detect objects → bounding boxes + class + confidence)
    ↓
DeepSORT (assign persistent track IDs using Kalman filter + Re-ID DNN)
    ↓
Analytics Engine (speed, direction, class counts)
    ↓
ROI Manager (check if objects are inside alert zones)
    ↓
Heatmap (accumulate centroid positions)
    ↓
Draw annotations (boxes, labels, trails, arrows, zones) on frame
    ↓
MJPEG stream → Browser via Flask
    ↓
Real-time Web Dashboard
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Screen shows "Press START" | Click **▶ Start** in the header; detection does not begin automatically |
| Upload failed | Make sure the server is running; check that the `logs/` folder exists |
| Seek bar not visible | It only appears for uploaded video files, not webcam |
| Camera not opening | Check `DEFAULT_SOURCE = 0` in config.py; try 1 or 2 |
| Slow FPS | Switch to YOLO11n (nano) model; reduce frame resolution |
| Tracking IDs keep resetting | Increase `DEEPSORT_MAX_AGE` in config.py |
| DeepSORT install fails | `pip install deep-sort-realtime --upgrade` |
| Port 5000 in use | Change `FLASK_PORT = 5001` in config.py |

---