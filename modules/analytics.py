# =============================================================================
#  modules/analytics.py — Speed, Direction, Object Counts & CSV Logging
# =============================================================================
#
#  For each tracked object this module:
#   1. Maintains a centroid history (deque of recent positions + timestamps)
#   2. Computes smoothed speed in px/s (and rough km/h)
#   3. Computes movement direction (N / NE / E … NW + angle in degrees)
#   4. Draws a direction arrow on the frame
#   5. Draws the motion trail (past N centroids)
#   6. Writes every detection to a CSV log file
#
# =============================================================================

import csv
import math
import os
import time
import logging
import numpy as np
import cv2
from collections import deque, defaultdict
from datetime import datetime

from config import (
    SPEED_HISTORY_LEN, TRAIL_LENGTH, SPEED_SCALE,
    ARROW_LENGTH, TRAIL_THICKNESS, LOG_DIR, CSV_FILENAME, CSV_COLUMNS,
)

logger = logging.getLogger(__name__)

# Compass direction lookup (8 cardinal + intercardinal directions)
_DIRECTIONS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def _angle_to_direction(dx: float, dy: float) -> str:
    """Convert (dx, dy) vector to compass direction string."""
    if dx == 0 and dy == 0:
        return "—"
    # atan2 gives angle from +x axis; y-axis is flipped in image coords
    angle = math.degrees(math.atan2(-dy, dx))   # negate dy (image y grows down)
    angle = (angle + 360) % 360
    idx   = int((angle + 22.5) / 45) % 8
    return _DIRECTIONS[idx]


def _angle_degrees(dx: float, dy: float) -> float:
    """Return angle in degrees (0=right, 90=up in image space)."""
    return math.degrees(math.atan2(-dy, dx))


class AnalyticsEngine:
    """
    Maintains per-track history and computes real-time analytics.

    Typical call per frame:
        engine.update(tracks, frame, frame_idx)
        engine.draw_trails_and_arrows(frame, tracks)
        summary = engine.get_summary()
    """

    def __init__(self):
        # track_id → deque of (cx, cy, timestamp)
        self._history:  dict = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
        # track_id → latest analytics dict
        self._stats:    dict = {}

        # Running totals
        self._class_counts:   dict = defaultdict(int)   # class → ever-seen count
        self._total_tracks:   int  = 0
        self._frame_count:    int  = 0
        self._session_start:  float = time.time()

        # FPS tracking
        self._fps_times: deque = deque(maxlen=60)
        self._fps: float = 0.0

        # CSV
        self._csv_path = os.path.join(LOG_DIR, CSV_FILENAME)
        self._init_csv()

        self._last_csv_time = 0.0

    # ── CSV ───────────────────────────────────────────────────────────────────

    def _init_csv(self):
        """Create CSV with header if it doesn't exist."""
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
        logger.info(f"CSV log: {self._csv_path}")

    def _log_to_csv(self, track: dict, frame_idx: int,
                    speed: float, direction: str, roi_zone: str):
        """Append one detection row to the CSV file."""
        x1, y1, x2, y2 = track["bbox"]
        cx, cy = track["centroid"]
        row = {
            "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "frame"     : frame_idx,
            "track_id"  : track["track_id"],
            "class"     : track["class_name"],
            "confidence": round(track["confidence"], 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": cx, "cy": cy,
            "speed_px_s": round(speed, 1),
            "direction" : direction,
            "roi_zone"  : roi_zone,
        }
        try:
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(row)
        except Exception as e:
            logger.warning(f"CSV write error: {e}")

    # ── Main Update ───────────────────────────────────────────────────────────

    def update(self, tracks: list, frame_idx: int,
               roi_labels: dict = None) -> None:
        """
        Called once per frame with the current list of confirmed tracks.

        Parameters
        ----------
        tracks     : list of track dicts from TrackerManager.update()
        frame_idx  : current frame number
        roi_labels : {track_id: zone_name} from RoiManager (optional)
        """
        now = time.time()

        # FPS
        self._fps_times.append(now)
        if len(self._fps_times) > 1:
            span = self._fps_times[-1] - self._fps_times[0]
            self._fps = (len(self._fps_times) - 1) / span if span > 0 else 0

        self._frame_count += 1
        current_class_counts: dict = defaultdict(int)
        write_csv = (now - self._last_csv_time) >= 0.25   # log max 4 rows/sec/track

        seen_tracks = set()

        for track in tracks:
            tid = track["track_id"]
            cx, cy = track["centroid"]
            seen_tracks.add(tid)

            # Append to history
            self._history[tid].append((cx, cy, now))

            # ── Speed & Direction ──────────────────────────────────────────
            speed_px_s = 0.0
            direction  = "—"

            hist = self._history[tid]
            if len(hist) >= 2:
                # Average over last SPEED_HISTORY_LEN entries for smoothing
                n = min(len(hist), SPEED_HISTORY_LEN)
                pts = list(hist)[-n:]
                dx_total = pts[-1][0] - pts[0][0]
                dy_total = pts[-1][1] - pts[0][1]
                dt = pts[-1][2] - pts[0][2]
                if dt > 0:
                    dist = math.sqrt(dx_total**2 + dy_total**2)
                    speed_px_s = dist / dt           # pixels per second
                direction = _angle_to_direction(dx_total, dy_total)

            speed_kmh = speed_px_s * SPEED_SCALE    # rough km/h

            # ── Update Counters ────────────────────────────────────────────
            cls = track["class_name"]
            current_class_counts[cls] += 1
            if tid not in self._stats:              # new track
                self._class_counts[cls] += 1
                self._total_tracks += 1

            roi_zone = (roi_labels or {}).get(tid, "")

            self._stats[tid] = {
                "speed_px_s": speed_px_s,
                "speed_kmh" : speed_kmh,
                "direction" : direction,
                "class"     : cls,
                "roi_zone"  : roi_zone,
            }

            # ── CSV Logging ────────────────────────────────────────────────
            if write_csv:
                self._log_to_csv(track, frame_idx, speed_px_s, direction, roi_zone)

        if write_csv:
            self._last_csv_time = now

        self._current_class_counts = dict(current_class_counts)
        self._current_track_count  = len(tracks)

        # Prune disappeared tracks from history (keep last 300 frames worth)
        if frame_idx % 300 == 0:
            dead = set(self._history.keys()) - seen_tracks
            for tid in dead:
                if tid not in [t["track_id"] for t in tracks]:
                    # keep a bit of history for trails that may still be drawn
                    pass    # deque with maxlen auto-prunes

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_trails_and_arrows(self, frame: np.ndarray, tracks: list) -> None:
        """
        Draw motion trails and direction arrows on the frame (in-place).

        Trail     : fading line of past centroid positions
        Arrow     : direction vector arrow originating from centroid
        """
        for track in tracks:
            tid   = track["track_id"]
            color = track["color"]
            hist  = list(self._history.get(tid, []))
            stats = self._stats.get(tid, {})

            # ── Trail ──────────────────────────────────────────────────────
            if len(hist) >= 2:
                pts = [(int(h[0]), int(h[1])) for h in hist]
                n   = len(pts)
                for i in range(1, n):
                    # Fade the trail: older segments are more transparent
                    alpha   = i / n
                    t_color = tuple(int(c * alpha) for c in color)
                    thickness = max(1, int(TRAIL_THICKNESS * alpha))
                    cv2.line(frame, pts[i-1], pts[i], t_color, thickness)

            # ── Direction Arrow ────────────────────────────────────────────
            if len(hist) >= SPEED_HISTORY_LEN // 2:
                n   = min(len(hist), SPEED_HISTORY_LEN)
                pts_s = list(hist)[-n:]
                dx = pts_s[-1][0] - pts_s[0][0]
                dy = pts_s[-1][1] - pts_s[0][1]
                mag = math.sqrt(dx**2 + dy**2)
                if mag > 3:                           # only draw if actually moving
                    scale = ARROW_LENGTH / mag
                    cx, cy = track["centroid"]
                    ex = int(cx + dx * scale)
                    ey = int(cy + dy * scale)
                    cv2.arrowedLine(frame, (cx, cy), (ex, ey),
                                    color, 2, tipLength=0.4)

    def draw_label(self, frame: np.ndarray, track: dict) -> None:
        """
        Draw the bounding box label: class | ID | speed | direction.
        Called from app.py's draw loop.
        """
        tid   = track["track_id"]
        stats = self._stats.get(tid, {})
        x1, y1, x2, y2 = track["bbox"]
        color = track["color"]

        spd   = stats.get("speed_px_s", 0.0)
        dirn  = stats.get("direction", "—")
        cls   = track["class_name"]
        conf  = track["confidence"]

        label_top    = f"{cls} #{tid}  {conf:.0%}"
        label_bottom = f"{spd:.0f}px/s  {dirn}"

        for i, text in enumerate([label_top, label_bottom]):
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            ty = max(y1 - (i+1)*18, 0)
            cv2.rectangle(frame, (x1, ty - th - 2), (x1 + tw + 4, ty + 2), color, -1)
            cv2.putText(frame, text, (x1 + 2, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    # ── Summary ───────────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Return a JSON-serialisable summary dict for the /api/analytics endpoint."""
        elapsed = time.time() - self._session_start
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)

        return {
            "fps"              : round(self._fps, 1),
            "current_count"   : self._current_track_count,
            "total_tracked"   : self._total_tracks,
            "frame_count"     : self._frame_count,
            "uptime"          : f"{h:02d}:{m:02d}:{s:02d}",
            "class_counts_now": self._current_class_counts,
            "class_counts_all": dict(self._class_counts),
            "csv_path"        : self._csv_path,
        }

    def get_track_stats(self, track_id: int) -> dict:
        return self._stats.get(track_id, {})

    def reset(self):
        self._history.clear()
        self._stats.clear()
        self._class_counts.clear()
        self._total_tracks   = 0
        self._frame_count    = 0
        self._session_start  = time.time()
        self._fps_times.clear()
        self._fps = 0.0
        self._init_csv()
        logger.info("Analytics reset.")
