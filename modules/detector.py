# =============================================================================
#  modules/detector.py — YOLOv8 Object Detection Wrapper
# =============================================================================
#
#  Wraps Ultralytics YOLOv8 to provide:
#   - Hot-swappable model loading (n / s / m / l / x)
#   - Per-frame detection with class + confidence filtering
#   - Returns clean detection dicts the rest of the pipeline uses
#
# =============================================================================

import numpy as np
import cv2
from ultralytics import YOLO
from config import COCO_CLASSES, DEFAULT_MODEL, DEFAULT_CONFIDENCE, DEFAULT_IOU
import logging

logger = logging.getLogger(__name__)


def _generate_colors(n: int):
    """
    Generate n visually distinct BGR colors using the golden-angle HSV trick.
    Each COCO class index always maps to the same color across the whole app.
    """
    colors = []
    for i in range(n):
        hue        = int((i * 137.508) % 180)           # golden angle spacing
        color_hsv  = np.uint8([[[hue, 220, 210]]])
        color_bgr  = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color_bgr))
    return colors


# Pre-baked palette — same color for a class no matter when it's detected
CLASS_COLORS = _generate_colors(len(COCO_CLASSES))


class Detector:
    """
    YOLOv8 detection wrapper.

    Usage:
        detector = Detector("yolov8n.pt")
        detections = detector.detect(frame, confidence=0.4, allowed_classes=None)

    Returns a list of dicts:
        {
          "bbox"      : (x1, y1, x2, y2),   # pixel coords
          "centroid"  : (cx, cy),
          "class_id"  : int,
          "class_name": str,
          "confidence": float,
          "color"     : (B, G, R),
          # DeepSORT raw format: ([x,y,w,h], conf, class_name)
          "deepsort"  : ([x, y, w, h], float, str),
        }
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name    = None
        self.model         = None
        self.load_model(model_name)

    # ── Model Management ──────────────────────────────────────────────────────

    def load_model(self, model_name: str):
        """Load (or hot-swap) a YOLOv8 model by filename."""
        if model_name == self.model_name:
            return                              # already loaded, skip
        try:
            logger.info(f"Loading YOLO model: {model_name}")
            self.model      = YOLO(model_name)
            self.model_name = model_name
            # Warm up with a dummy frame so first real frame isn't slow
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)
            logger.info(f"✅ Model '{model_name}' ready.")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(
        self,
        frame:           np.ndarray,
        confidence:      float      = DEFAULT_CONFIDENCE,
        iou_threshold:   float      = DEFAULT_IOU,
        allowed_classes: list       = None,   # None = all classes
    ) -> list:
        """
        Run YOLOv8 on a single BGR frame.

        Parameters
        ----------
        frame           : OpenCV BGR image
        confidence      : minimum detection confidence (0–1)
        iou_threshold   : NMS overlap threshold
        allowed_classes : list of class names to keep; None = keep everything

        Returns
        -------
        List of detection dicts (see class docstring above).
        """
        if self.model is None:
            return []

        try:
            results = self.model.predict(
                frame,
                conf    = confidence,
                iou     = iou_threshold,
                verbose = False,
            )
        except Exception as e:
            logger.warning(f"YOLO predict error: {e}")
            return []

        detections = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for box in boxes:
            class_id   = int(box.cls[0])
            conf       = float(box.conf[0])
            class_name = (COCO_CLASSES[class_id]
                          if class_id < len(COCO_CLASSES) else "unknown")

            # Apply class filter
            if allowed_classes and class_name not in allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w  = x2 - x1
            h  = y2 - y1
            cx = x1 + w // 2
            cy = y1 + h // 2

            detections.append({
                "bbox"      : (x1, y1, x2, y2),
                "centroid"  : (cx, cy),
                "class_id"  : class_id,
                "class_name": class_name,
                "confidence": round(conf, 3),
                "color"     : CLASS_COLORS[class_id % len(CLASS_COLORS)],
                # DeepSORT expects ([x,y,w,h], conf, class_label)
                "deepsort"  : ([x1, y1, w, h], conf, class_name),
            })

        return detections
