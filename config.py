# =============================================================================
#  config.py — Global Configuration for Object Detection & Tracking System
# =============================================================================

import os

# ── Directory Paths ──────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
LOG_DIR   = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")   # cached YOLO weights go here

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Available YOLO Models ────────────────────────────────────────────────────
#
#  TWO GENERATIONS SUPPORTED:
#
#  YOLO11  (2024) ★ RECOMMENDED
#    - Latest Ultralytics model released Oct 2024
#    - 22% fewer parameters than YOLOv8 → faster on CPU (your laptop)
#    - Higher accuracy at the same speed
#    - Fully compatible — same API, zero code changes needed
#    - Model files are slightly SMALLER than v8 equivalents
#
#  YOLOv8  (2023) — Stable fallback
#    - Previous generation, battle-tested
#    - Slightly larger model files
#    - Use if YOLO11 gives any compatibility issues
#
#  HOW MODELS ARE DOWNLOADED:
#    First time you select a model, Ultralytics auto-downloads it.
#    After that it's cached locally and loads instantly.
#
#  SIZE COMPARISON:
#    YOLO11n  ~5MB  vs  YOLOv8n  ~6MB
#    YOLO11s ~19MB  vs  YOLOv8s ~22MB
#    YOLO11m ~38MB  vs  YOLOv8m ~52MB  ← 27% smaller!
#    YOLO11l ~49MB  vs  YOLOv8l ~87MB  ← 44% smaller!
#    YOLO11x ~109MB vs  YOLOv8x ~131MB
#
YOLO_MODELS = {

    # ── YOLO11 — 2024 Latest (Recommended) ───────────────────────────────
    "YOLO11n — Nano    (Fastest,   ~5MB)  LATEST" : "yolo11n.pt",
    "YOLO11s — Small   (Fast,     ~19MB)  LATEST" : "yolo11s.pt",
    "YOLO11m — Medium  (Balanced, ~38MB)  LATEST" : "yolo11m.pt",
    "YOLO11l — Large   (Accurate, ~49MB)  LATEST" : "yolo11l.pt",
    "YOLO11x — XLarge  (Best Acc,~109MB)  LATEST" : "yolo11x.pt",

    # ── YOLOv8 — 2023 Stable Fallback ────────────────────────────────────
    "YOLOv8n — Nano    (Fastest,   ~6MB)  STABLE" : "yolov8n.pt",
    "YOLOv8s — Small   (Fast,     ~22MB)  STABLE" : "yolov8s.pt",
    "YOLOv8m — Medium  (Balanced, ~52MB)  STABLE" : "yolov8m.pt",
    "YOLOv8l — Large   (Accurate, ~87MB)  STABLE" : "yolov8l.pt",
    "YOLOv8x — XLarge  (Best Acc,~131MB)  STABLE" : "yolov8x.pt",
}

# ── Default Model ─────────────────────────────────────────────────────────────
#  "yolo11n.pt" → fastest, best for laptops without GPU  ← RECOMMENDED
#  "yolo11s.pt" → slightly slower but noticeably better accuracy
#  "yolov8n.pt" → use this if you face any YOLO11 issues
DEFAULT_MODEL      = "yolo11n.pt"

DEFAULT_CONFIDENCE = 0.40       # Min confidence to accept a detection (0–1)
DEFAULT_IOU        = 0.45       # Non-max suppression overlap threshold

# ── Video Source ─────────────────────────────────────────────────────────────
DEFAULT_SOURCE = 0              # 0 = webcam; set to file path for video file
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
TARGET_FPS     = 30

# ── DeepSORT Tracking Config ─────────────────────────────────────────────────
DEEPSORT_MAX_AGE        = 60    # Frames to keep a track alive without detection
DEEPSORT_N_INIT         = 3     # Detections needed to confirm a new track
DEEPSORT_MAX_IOU        = 0.7   # Max IoU for NMS in tracker
DEEPSORT_MAX_COSINE     = 0.4   # Max cosine distance for appearance matching

# ── Analytics ────────────────────────────────────────────────────────────────
SPEED_HISTORY_LEN  = 10         # Frames to average for speed smoothing
TRAIL_LENGTH       = 30         # Number of past positions to draw as trail
SPEED_SCALE        = 0.05       # px/frame → km/h (rough, scene-dependent)
ANALYTICS_INTERVAL = 1.0        # Seconds between analytics snapshots to CSV

# ── Heatmap ──────────────────────────────────────────────────────────────────
HEATMAP_ALPHA  = 0.45           # Blend ratio: 0=original, 1=pure heatmap
HEATMAP_DECAY  = 0.995          # How fast heatmap fades (lower = faster fade)
HEATMAP_RADIUS = 30             # Gaussian blur radius for each detection point

# ── CSV Logging ──────────────────────────────────────────────────────────────
CSV_FILENAME = "detections.csv"
CSV_COLUMNS  = [
    "timestamp", "frame", "track_id", "class", "confidence",
    "x1", "y1", "x2", "y2", "cx", "cy",
    "speed_px_s", "direction", "roi_zone"
]

# ── ROI Zones ────────────────────────────────────────────────────────────────
MAX_ROI_ZONES = 5               # Maximum number of ROI zones allowed

# ── Drawing Styles ───────────────────────────────────────────────────────────
BOX_THICKNESS  = 2
FONT_SCALE     = 0.55
FONT_THICKNESS = 2
TRAIL_THICKNESS= 2
ROI_ALPHA      = 0.15           # ROI fill transparency
ARROW_LENGTH   = 30             # Direction arrow length in pixels

# ── COCO Class Names (80 categories) ─────────────────────────────────────────
#  Both YOLO11 and YOLOv8 are trained on the same COCO dataset
#  so these 80 class names work for BOTH model generations.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# ── Flask ─────────────────────────────────────────────────────────────────────
FLASK_HOST  = "0.0.0.0"
FLASK_PORT  = 5000
FLASK_DEBUG = False
SECRET_KEY  = "visor-od-tracking-2024"