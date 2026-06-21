# =============================================================================
#  modules/tracker.py — DeepSORT Multi-Object Tracking Wrapper
# =============================================================================
#
#  DeepSORT (Deep Simple Online and Realtime Tracking) improves on basic
#  centroid tracking by combining:
#    1. Kalman filter   — predicts where an object will be next frame
#    2. Appearance DNN  — matches tracks using visual features (re-ID)
#    3. IoU matching    — confirms matches using bounding box overlap
#
#  This wrapper translates between the app's detection format and
#  deep_sort_realtime's API, and enriches tracks with class information.
#
# =============================================================================

import numpy as np
import logging
from collections import defaultdict

from deep_sort_realtime.deepsort_tracker import DeepSort
from config import (
    DEEPSORT_MAX_AGE, DEEPSORT_N_INIT,
    DEEPSORT_MAX_IOU, DEEPSORT_MAX_COSINE,
)

logger = logging.getLogger(__name__)


class TrackerManager:
    """
    Wraps DeepSORT and returns enriched track dicts compatible with
    the rest of the pipeline.

    Each track dict:
        {
            "track_id"  : int,
            "bbox"      : (x1, y1, x2, y2),
            "centroid"  : (cx, cy),
            "class_name": str,
            "class_id"  : int,
            "confidence": float,
            "color"     : (B, G, R),
            "age"       : int,      # frames this track has been alive
        }
    """

    def __init__(self):
        self._init_tracker()
        # Keep per-track class info because DeepSORT may lose it
        self._track_classes: dict = {}     # track_id → class_name
        self._track_confs:   dict = {}     # track_id → confidence
        self._track_colors:  dict = {}     # track_id → color
        self._track_ages:    dict = defaultdict(int)

    def _init_tracker(self):
        """Create a fresh DeepSORT instance."""
        self.tracker = DeepSort(
            max_age            = DEEPSORT_MAX_AGE,
            n_init             = DEEPSORT_N_INIT,
            nms_max_overlap    = DEEPSORT_MAX_IOU,
            max_cosine_distance= DEEPSORT_MAX_COSINE,
            nn_budget          = 100,          # max feature vectors stored per class
            override_track_class=None,
            embedder           = "mobilenet",  # lightweight re-ID model
            half               = False,
            bgr                = True,         # our frames are BGR (OpenCV default)
        )

    def reset(self):
        """Reset tracker — call when source changes."""
        self._init_tracker()
        self._track_classes.clear()
        self._track_confs.clear()
        self._track_colors.clear()
        self._track_ages.clear()
        logger.info("Tracker reset.")

    def update(self, detections: list, frame: np.ndarray) -> list:
        """
        Update tracker with new detections and return confirmed tracks.

        Parameters
        ----------
        detections : list of detection dicts from Detector.detect()
        frame      : BGR frame (used by re-ID embedder)

        Returns
        -------
        List of track dicts.
        """
        if not detections:
            # Still update tracker with empty list so Kalman filters step forward
            raw = []
        else:
            # Build the raw format DeepSORT expects:
            # ([left, top, width, height], confidence, class_label)
            raw = [d["deepsort"] for d in detections]

            # Cache class/confidence/color info keyed by detection index
            # (DeepSORT will give back track_id later, not detection index)
            for d in detections:
                # We don't know track_id yet, so we cache by bbox hash
                key = _bbox_key(d["bbox"])
                self._pending_class[key] = (
                    d["class_name"], d["confidence"], d["color"], d["class_id"]
                )

        try:
            tracks = self.tracker.update_tracks(raw, frame=frame)
        except Exception as e:
            logger.warning(f"DeepSORT update error: {e}")
            return []

        results = []
        active_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            active_ids.add(tid)
            self._track_ages[tid] += 1

            # Get bounding box (left, top, right, bottom)
            try:
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
            except Exception:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Retrieve class info from our cache (persists across frames)
            # Try to update from pending dict first
            best_key = _find_nearest_key(self._pending_class, (x1,y1,x2,y2))
            if best_key and best_key in self._pending_class:
                class_name, conf, color, class_id = self._pending_class[best_key]
                self._track_classes[tid] = class_name
                self._track_confs[tid]   = conf
                self._track_colors[tid]  = color

            results.append({
                "track_id"  : tid,
                "bbox"      : (x1, y1, x2, y2),
                "centroid"  : (cx, cy),
                "class_name": self._track_classes.get(tid, "object"),
                "confidence": self._track_confs.get(tid, 0.0),
                "color"     : self._track_colors.get(tid, (0, 255, 100)),
                "age"       : self._track_ages[tid],
            })

        # Clean up ages for tracks that disappeared
        dead_ids = set(self._track_ages.keys()) - active_ids
        for tid in dead_ids:
            if self._track_ages[tid] > DEEPSORT_MAX_AGE * 2:
                self._track_ages.pop(tid, None)
                self._track_classes.pop(tid, None)
                self._track_confs.pop(tid, None)
                self._track_colors.pop(tid, None)

        self._pending_class.clear()
        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    @property
    def _pending_class(self):
        if not hasattr(self, "_pc"):
            self._pc = {}
        return self._pc


def _bbox_key(bbox) -> str:
    return f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"


def _find_nearest_key(cache: dict, bbox: tuple) -> str:
    """
    Find the cache entry whose bbox is closest to the given bbox.
    Used to associate DeepSORT output back to our detection metadata.
    """
    if not cache:
        return None
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)//2, (y1+y2)//2
    best_key  = None
    best_dist = float("inf")
    for key in cache:
        try:
            kx1, ky1, kx2, ky2 = map(int, key.split(","))
            kx, ky = (kx1+kx2)//2, (ky1+ky2)//2
            dist = (kx - cx)**2 + (ky - cy)**2
            if dist < best_dist:
                best_dist = dist
                best_key  = key
        except Exception:
            continue
    return best_key
