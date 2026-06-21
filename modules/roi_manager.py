# =============================================================================
#  modules/roi_manager.py — Region of Interest (ROI) Zone Manager
# =============================================================================
#
#  Lets the user define up to 5 rectangular alert zones via the web UI.
#  For every frame the manager:
#   1. Checks which tracked objects fall inside each zone
#   2. Returns a {track_id: zone_name} label dict for analytics / CSV
#   3. Draws the zones (with counts + alert colours) on the frame
#
# =============================================================================

import cv2
import numpy as np
import threading
import logging

from config import MAX_ROI_ZONES, ROI_ALPHA

logger = logging.getLogger(__name__)

# Distinct zone colours in BGR (cycling through these)
_ZONE_COLORS = [
    (0,   200, 255),    # amber
    (0,   255, 128),    # neon green
    (255,  80,  80),    # red
    (200, 100, 255),    # purple
    (255, 220,   0),    # yellow
]


class RoiManager:
    """
    Manages a set of named rectangular ROI zones.

    A zone is defined as:
        {
            "name"  : str,               e.g. "Entry Gate"
            "x1"    : int,               pixel coords
            "y1"    : int,
            "x2"    : int,
            "y2"    : int,
            "color" : (B, G, R),         auto-assigned
            "count" : int,               updated each frame
        }
    """

    def __init__(self):
        self._zones: list  = []
        self._lock = threading.Lock()

    # ── Zone Management ───────────────────────────────────────────────────────

    def set_zones(self, zone_list: list) -> None:
        """
        Replace the current zone list.

        zone_list : list of dicts with keys: name, x1, y1, x2, y2
                    (as sent from the front end)
        """
        with self._lock:
            self._zones = []
            for i, z in enumerate(zone_list[:MAX_ROI_ZONES]):
                self._zones.append({
                    "name"  : z.get("name", f"Zone {i+1}"),
                    "x1"    : int(z.get("x1", 0)),
                    "y1"    : int(z.get("y1", 0)),
                    "x2"    : int(z.get("x2", 100)),
                    "y2"    : int(z.get("y2", 100)),
                    "color" : _ZONE_COLORS[i % len(_ZONE_COLORS)],
                    "count" : 0,
                })
            logger.info(f"ROI zones set: {[z['name'] for z in self._zones]}")

    def get_zones_json(self) -> list:
        """Return zone definitions (without OpenCV color tuples) as JSON-safe list."""
        with self._lock:
            result = []
            for z in self._zones:
                result.append({
                    "name"  : z["name"],
                    "x1": z["x1"], "y1": z["y1"],
                    "x2": z["x2"], "y2": z["y2"],
                    "count" : z["count"],
                    "color_hex": "#{:02x}{:02x}{:02x}".format(
                        z["color"][2], z["color"][1], z["color"][0]  # BGR → RGB
                    ),
                })
            return result

    def clear_zones(self):
        with self._lock:
            self._zones = []

    # ── Per-Frame Processing ──────────────────────────────────────────────────

    def process(self, tracks: list) -> dict:
        """
        Determine which ROI zone (if any) each track currently occupies.

        Parameters
        ----------
        tracks : list of track dicts from TrackerManager

        Returns
        -------
        dict : {track_id: zone_name}  — only for tracks inside a zone
        roi_alerts : list of zone summary dicts
        """
        roi_labels: dict = {}

        with self._lock:
            # Reset per-zone counts
            for z in self._zones:
                z["count"] = 0

            for track in tracks:
                cx, cy = track["centroid"]
                for z in self._zones:
                    if z["x1"] <= cx <= z["x2"] and z["y1"] <= cy <= z["y2"]:
                        roi_labels[track["track_id"]] = z["name"]
                        z["count"] += 1
                        break   # one zone per track

        return roi_labels

    def draw_zones(self, frame: np.ndarray) -> None:
        """
        Draw all ROI zones on the frame (in-place).
        Each zone shows: coloured border + semi-transparent fill + label + count.
        """
        with self._lock:
            zones = list(self._zones)

        overlay = frame.copy()

        for z in zones:
            x1, y1, x2, y2 = z["x1"], z["y1"], z["x2"], z["y2"]
            color  = z["color"]
            count  = z["count"]
            name   = z["name"]

            # ── Semi-transparent fill ──────────────────────────────────────
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            # ── Dashed border effect (draw 4 corner rectangles) ───────────
            t = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, t)

            # Alert: thicker border + brighter fill when objects are present
            if count > 0:
                cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), color, 3)

            # ── Label banner ──────────────────────────────────────────────
            label = f"  {name}  [{count}]  "
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0) if sum(color) > 400 else (255, 255, 255),
                        2)

        # Blend fill overlay
        cv2.addWeighted(overlay, ROI_ALPHA, frame, 1 - ROI_ALPHA, 0, frame)

    def get_alert_summary(self) -> list:
        """Return list of zone alert dicts for the analytics endpoint."""
        with self._lock:
            return [
                {"zone": z["name"], "count": z["count"]}
                for z in self._zones
            ]
