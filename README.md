# 🎯Advanced Object Detection & Tracking System

A professional-grade, browser-based real-time object detection and tracking system
built with **YOLOv8 + DeepSORT + Flask**.

---

## 📁 Project Structure

```
CodeAlpha_ObjectDetection_Advanced/
│
├── app.py                     ← Flask server + video processing thread
├── config.py                  ← All settings & constants (edit here!)
├── requirements.txt           ← Python dependencies
│
├── modules/
│   ├── detector.py            ← YOLOv8 detection wrapper
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
│   └── js/main.js             ← Frontend controls & ROI canvas
│
└── logs/
    └── detections.csv         ← Auto-generated detection log
```

---

## ⚡ Quick Setup (Step by Step)

### Step 1 — Install Python 3.9+
Download from: https://python.org/downloads  
✅ Check **"Add Python to PATH"** during installation

### Step 2 — Install dependencies
Open a terminal in the project folder and run:
```bash
pip install -r requirements.txt
```
> This installs Flask, YOLOv8, DeepSORT, OpenCV, and NumPy.
> YOLOv8 model weights (~6MB) download automatically on first run.

### Step 3 — Run the server
```bash
python app.py
```

### Step 4 — Open the dashboard
Open your browser and go to:
```
http://localhost:5000
```

---

## 🎮 Features & How to Use

### 🔍 Multiple YOLO Models
In the **Controls** tab, choose from:
| Model | Speed | Accuracy | Size |
|-------|-------|----------|------|
| YOLOv8n (Nano)  | ⚡⚡⚡⚡⚡ | ⭐⭐ | ~6MB |
| YOLOv8s (Small) | ⚡⚡⚡⚡  | ⭐⭐⭐ | ~22MB |
| YOLOv8m (Medium)| ⚡⚡⚡    | ⭐⭐⭐⭐ | ~52MB |
| YOLOv8l (Large) | ⚡⚡      | ⭐⭐⭐⭐⭐ | ~87MB |
| YOLOv8x (XLarge)| ⚡        | ⭐⭐⭐⭐⭐ | ~131MB |

Hot-swap between models without restarting!

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
- **Video File**: Upload an .mp4 / .avi file
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
Webcam/Video
    ↓
OpenCV (frame capture)
    ↓
YOLOv8 (detect objects → bounding boxes + class + confidence)
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
| Camera not opening | Check `DEFAULT_SOURCE = 0` in config.py; try 1 or 2 |
| Slow FPS | Switch to YOLOv8n model; reduce frame resolution |
| Tracking IDs keep resetting | Increase `DEEPSORT_MAX_AGE` in config.py |
| DeepSORT install fails | `pip install deep-sort-realtime --upgrade` |
| Port 5000 in use | Change `FLASK_PORT = 5001` in config.py |

---