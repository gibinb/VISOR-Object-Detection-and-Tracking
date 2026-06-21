import cv2
import threading
import time
import os
import logging
import numpy as np

from flask import (Flask, Response, render_template,
                   request, jsonify, send_file)

import config
from modules.detector    import Detector, CLASS_COLORS
from modules.tracker     import TrackerManager
from modules.analytics   import AnalyticsEngine
from modules.heatmap     import HeatmapGenerator
from modules.roi_manager import RoiManager

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("app")

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = config.SECRET_KEY


# =============================================================================
#  VideoProcessor — Background Thread
# =============================================================================

class VideoProcessor:
    """
    States
    ──────
      STOPPED  → camera closed, thread idles, standby screen shown
      RUNNING  → camera open, detection + tracking active
      PAUSED   → camera open, pipeline frozen, last frame shown

    Seek / Scrub (video files only)
    ────────────────────────────────
      self._seek_to_frame  — if set to an int, the inner loop jumps to
                             that frame position then clears the value.
      self._video_info     — dict with total_frames / fps / duration_s
                             populated once the source is opened.
      self._current_frame  — updated every iteration so the browser can
                             poll /api/video_progress for the seek bar.
    """

    STATE_STOPPED = "stopped"
    STATE_RUNNING = "running"
    STATE_PAUSED  = "paused"

    def __init__(self):
        # Pipeline modules
        self.detector  = Detector(config.DEFAULT_MODEL)
        self.tracker   = TrackerManager()
        self.analytics = AnalyticsEngine()
        self.heatmap   = HeatmapGenerator()
        self.roi       = RoiManager()

        # Shared output (guarded by _lock)
        self._lock          = threading.Lock()
        self._frame_out     = None
        self._heatmap_out   = None
        self._analytics_out = {}

        # Processing state
        self._state      = self.STATE_STOPPED
        self._state_lock = threading.Lock()

        # Settings
        self.model_name      = config.DEFAULT_MODEL
        self.confidence      = config.DEFAULT_CONFIDENCE
        self.iou_threshold   = config.DEFAULT_IOU
        self.source          = config.DEFAULT_SOURCE
        self.allowed_classes = None
        self.show_heatmap    = False
        self.show_trails     = True
        self.show_roi        = True
        self.tracking_on     = True

        # ── Seek / scrub ──────────────────────────────────────────────────
        self._seek_to_frame  = None   # set by seek() → picked up in loop
        self._seek_lock      = threading.Lock()
        self._current_frame  = 0      # updated every frame (for progress bar)
        self._video_info     = {}     # total_frames, fps, duration_s, is_file

        # Thread control
        self._stop_evt = threading.Event()
        self._restart  = False
        self._thread   = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        self._write_standby_frame()
        logger.info("VideoProcessor started (state=STOPPED)")

    # =========================================================================
    #  State helpers
    # =========================================================================

    @property
    def state(self):
        with self._state_lock:
            return self._state

    def _set_state(self, new_state):
        with self._state_lock:
            old = self._state
            self._state = new_state
        logger.info(f"State: {old} → {new_state}")

    def start(self):
        if self.state == self.STATE_RUNNING:
            return
        if self.state == self.STATE_PAUSED:
            self._set_state(self.STATE_RUNNING)
            return
        self._set_state(self.STATE_RUNNING)

    def pause(self):
        if self.state == self.STATE_RUNNING:
            self._set_state(self.STATE_PAUSED)

    def resume(self):
        if self.state == self.STATE_PAUSED:
            self._set_state(self.STATE_RUNNING)

    def stop(self):
        if self.state != self.STATE_STOPPED:
            self._set_state(self.STATE_STOPPED)
            self._restart = True
            self._video_info = {}
            self._current_frame = 0
            self._write_standby_frame()

    # =========================================================================
    #  Seek / Scrub
    # =========================================================================

    def seek(self, position: float):
        """
        Seek to a position in the video.

        Parameters
        ----------
        position : float between 0.0 and 1.0  (fraction of total duration)
                   OR an int (absolute frame number)
        """
        info = self._video_info
        if not info.get("is_file"):
            return   # can't seek on a webcam

        total = info.get("total_frames", 0)
        if total <= 0:
            return

        # Convert fraction → frame number
        if isinstance(position, float) and 0.0 <= position <= 1.0:
            target_frame = int(position * total)
        else:
            target_frame = int(position)

        target_frame = max(0, min(target_frame, total - 1))

        with self._seek_lock:
            self._seek_to_frame = target_frame
        logger.info(f"Seek requested → frame {target_frame} / {total}")

    def get_video_progress(self) -> dict:
        """Return current playback position info for the seek bar."""
        info  = self._video_info
        total = info.get("total_frames", 0)
        cur   = self._current_frame
        fps   = info.get("fps", 25) or 25

        fraction   = (cur / total) if total > 0 else 0.0
        cur_sec    = cur / fps
        total_sec  = total / fps

        return {
            "is_file"     : info.get("is_file", False),
            "current_frame": cur,
            "total_frames" : total,
            "fraction"     : round(fraction, 5),
            "current_time" : _fmt_time(cur_sec),
            "total_time"   : _fmt_time(total_sec),
            "current_sec"  : round(cur_sec, 2),
            "total_sec"    : round(total_sec, 2),
        }

    # =========================================================================
    #  Main thread loop
    # =========================================================================

    def _run(self):
        while not self._stop_evt.is_set():
            if self.state == self.STATE_RUNNING:
                try:
                    self._process_source()
                except Exception as e:
                    logger.error(f"Processing error: {e}", exc_info=True)
                    self._set_state(self.STATE_STOPPED)
                    self._write_error_frame(str(e))
                    time.sleep(2)
            else:
                time.sleep(0.1)

    def _process_source(self):
        source = self.source
        logger.info(f"Opening source: {source}")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Cannot open source: {source}")
            self._write_error_frame(f"Cannot open source: {source}")
            self._set_state(self.STATE_STOPPED)
            time.sleep(3)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

        # ── Collect video info ─────────────────────────────────────────────
        is_file     = isinstance(source, str) and os.path.isfile(str(source))
        total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
        src_fps     = cap.get(cv2.CAP_PROP_FPS) or 25
        duration_s  = (total_frames / src_fps) if (is_file and src_fps) else 0

        self._video_info = {
            "is_file"     : is_file,
            "total_frames": total_frames,
            "fps"         : src_fps,
            "duration_s"  : duration_s,
        }
        logger.info(f"Video info: {self._video_info}")

        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
        self.heatmap.set_size(fh, fw)

        self.tracker.reset()
        self.analytics.reset()
        self.heatmap.reset()
        self._restart       = False
        self._current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame_idx = 0
        last_vis  = None

        while not self._stop_evt.is_set() and not self._restart:

            # ── Handle pending seek ────────────────────────────────────────
            with self._seek_lock:
                seek_target = self._seek_to_frame
                self._seek_to_frame = None

            if seek_target is not None and is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
                self._current_frame = seek_target
                # Reset tracker so IDs don't go crazy after seeking
                self.tracker.reset()
                self.analytics.reset()
                frame_idx = seek_target

            current_state = self.state

            if current_state == self.STATE_STOPPED:
                break

            # ── PAUSED ────────────────────────────────────────────────────
            if current_state == self.STATE_PAUSED:
                if last_vis is not None:
                    paused_frame = last_vis.copy()
                    self._draw_pause_overlay(paused_frame)
                    # Draw seek bar on paused frame too
                    if is_file and total_frames > 0:
                        _draw_seek_bar(paused_frame, self._current_frame,
                                       total_frames, src_fps)
                    _, jpg = cv2.imencode(".jpg", paused_frame,
                                          [cv2.IMWRITE_JPEG_QUALITY, 85])
                    with self._lock:
                        self._frame_out = jpg.tobytes()
                time.sleep(0.05)
                continue

            # ── RUNNING ───────────────────────────────────────────────────
            ret, frame = cap.read()

            if not ret:
                if is_file:
                    # Video ended — loop back to start
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._current_frame = 0
                    frame_idx = 0
                    self.tracker.reset()
                    self.analytics.reset()
                    continue
                else:
                    logger.warning("Frame read failed (webcam lost?).")
                    time.sleep(0.5)
                    continue

            # Update current frame counter
            self._current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Model hot-swap
            if self.detector.model_name != self.model_name:
                try:
                    self.detector.load_model(self.model_name)
                except Exception as e:
                    logger.error(f"Model load failed: {e}")

            # Detection
            detections = self.detector.detect(
                frame,
                confidence      = self.confidence,
                iou_threshold   = self.iou_threshold,
                allowed_classes = self.allowed_classes,
            )

            # Tracking
            if self.tracking_on:
                tracks = self.tracker.update(detections, frame)
            else:
                tracks = [
                    {
                        "track_id"  : i,
                        "bbox"      : d["bbox"],
                        "centroid"  : d["centroid"],
                        "class_name": d["class_name"],
                        "confidence": d["confidence"],
                        "color"     : d["color"],
                        "age"       : 1,
                    }
                    for i, d in enumerate(detections)
                ]

            # ROI
            roi_labels = {}
            if self.show_roi:
                roi_labels = self.roi.process(tracks)

            # Analytics
            self.analytics.update(tracks, frame_idx, roi_labels)

            # Heatmap
            centroids = [t["centroid"] for t in tracks]
            self.heatmap.update(centroids)

            # Annotate frame
            if self.show_heatmap:
                vis_frame = self.heatmap.blend(frame)
            else:
                vis_frame = frame.copy()

            if self.show_trails and self.tracking_on:
                self.analytics.draw_trails_and_arrows(vis_frame, tracks)

            for track in tracks:
                x1, y1, x2, y2 = track["bbox"]
                color = track["color"]
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                _draw_corner_marks(vis_frame, x1, y1, x2, y2, color)
                self.analytics.draw_label(vis_frame, track)

            if self.show_roi:
                self.roi.draw_zones(vis_frame)

            _draw_hud(vis_frame, self.analytics.get_summary(),
                      self.model_name, self.detector.model_name)

            # Draw seek bar overlay on video frame (file only)
            if is_file and total_frames > 0:
                _draw_seek_bar(vis_frame, self._current_frame,
                               total_frames, src_fps)

            last_vis = vis_frame

            _, jpg = cv2.imencode(".jpg", vis_frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            hm_img = self.heatmap.get_image()
            if hm_img is not None:
                _, hm_jpg = cv2.imencode(".jpg", hm_img,
                                          [cv2.IMWRITE_JPEG_QUALITY, 80])
            else:
                hm_jpg = jpg

            with self._lock:
                self._frame_out     = jpg.tobytes()
                self._heatmap_out   = hm_jpg.tobytes()
                self._analytics_out = self.analytics.get_summary()
                self._analytics_out["roi_alerts"] = self.roi.get_alert_summary()
                self._analytics_out["model"]      = self.detector.model_name
                self._analytics_out["source"]     = (
                    "webcam" if source == 0
                    else os.path.basename(str(source))
                )
                self._analytics_out["state"] = self.state

            frame_idx += 1

        cap.release()
        logger.info("VideoCapture released.")

    # =========================================================================
    #  Thread-safe getters
    # =========================================================================

    def get_frame(self):
        with self._lock:
            return self._frame_out

    def get_heatmap(self):
        with self._lock:
            return self._heatmap_out

    def get_analytics(self):
        with self._lock:
            d = dict(self._analytics_out)
            d["state"] = self.state
            return d

    # =========================================================================
    #  Settings
    # =========================================================================

    def apply_settings(self, data: dict):
        if "model" in data:
            self.model_name = data["model"]
        if "confidence" in data:
            self.confidence = float(data["confidence"])
        if "iou" in data:
            self.iou_threshold = float(data["iou"])
        if "classes" in data:
            classes = data["classes"]
            self.allowed_classes = classes if classes else None
        if "show_heatmap" in data:
            self.show_heatmap = bool(data["show_heatmap"])
        if "show_trails" in data:
            self.show_trails = bool(data["show_trails"])
        if "tracking_on" in data:
            self.tracking_on = bool(data["tracking_on"])
        if "show_roi" in data:
            self.show_roi = bool(data["show_roi"])
        if "source" in data:
            new_source = data["source"]
            if isinstance(new_source, str) and new_source.isdigit():
                new_source = int(new_source)
            if new_source != self.source:
                self.source   = new_source
                self._restart = True
                self._video_info    = {}
                self._current_frame = 0
        logger.info(f"Settings updated: {data}")

    # =========================================================================
    #  Special frames
    # =========================================================================

    def _write_standby_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for x in range(0, 640, 40):
            cv2.line(frame, (x, 0), (x, 480), (0, 25, 15), 1)
        for y in range(0, 480, 40):
            cv2.line(frame, (0, y), (640, y), (0, 25, 15), 1)

        cv2.circle(frame, (320, 185), 62, (0, 100, 50), 2)
        cv2.circle(frame, (320, 185), 48, (0, 70,  35), 1)
        pts = np.array([[300, 160], [300, 212], [352, 186]], np.int32)
        cv2.fillPoly(frame, [pts], (0, 200, 100))

        text_visor = "V I S O R"
        tw, _ = cv2.getTextSize(text_visor, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0]
        cv2.putText(frame, text_visor, ((640 - tw) // 2, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 110), 2)

        tag = "See Everything. Miss Nothing."
        tw2, _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
        cv2.putText(frame, tag, ((640 - tw2) // 2, 335),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 150, 75), 1)

        ins = "Press  START  to begin detection"
        tw3, _ = cv2.getTextSize(ins, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0]
        cv2.putText(frame, ins, ((640 - tw3) // 2, 378),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 130, 65), 1)

        _, jpg = cv2.imencode(".jpg", frame)
        with self._lock:
            self._frame_out = jpg.tobytes()

    def _write_error_frame(self, message: str):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "ERROR", (200, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 180), 5)
        cv2.putText(frame, message[:60], (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 200), 1)
        cv2.putText(frame, "Click START to retry",
                    (190, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 160), 1)
        _, jpg = cv2.imencode(".jpg", frame)
        with self._lock:
            self._frame_out = jpg.tobytes()

    @staticmethod
    def _draw_pause_overlay(frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h//2 - 36), (w, h//2 + 36), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (w//2 - 30, h//2 - 22),
                      (w//2 - 8,  h//2 + 22), (0, 220, 120), -1)
        cv2.rectangle(frame, (w//2 + 8,  h//2 - 22),
                      (w//2 + 30, h//2 + 22), (0, 220, 120), -1)
        cv2.putText(frame, "PAUSED", (w//2 + 40, h//2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 120), 2)

    def stop_thread(self):
        self._stop_evt.set()


# =============================================================================
#  Drawing helpers
# =============================================================================

def _fmt_time(seconds: float) -> str:
    """Convert seconds → MM:SS string."""
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


def _draw_seek_bar(frame, current_frame: int, total_frames: int, fps: float):
    """
    Draw a slim progress bar + time stamps at the bottom of the frame.
    This is the OpenCV overlay — a matching HTML scrubber is in index.html.
    """
    h, w = frame.shape[:2]
    bar_h    = 4
    bar_y    = h - 28
    pad      = 12
    bar_w    = w - pad * 2

    fraction = current_frame / total_frames if total_frames > 0 else 0
    filled   = int(bar_w * fraction)

    # Dark background strip
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 38), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Track (grey)
    cv2.rectangle(frame, (pad, bar_y), (pad + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    # Fill (green)
    if filled > 0:
        cv2.rectangle(frame, (pad, bar_y), (pad + filled, bar_y + bar_h),
                      (0, 220, 110), -1)
    # Thumb dot
    thumb_x = pad + filled
    cv2.circle(frame, (thumb_x, bar_y + bar_h // 2), 6, (0, 255, 128), -1)

    # Time text
    cur_time   = _fmt_time(current_frame / fps if fps else 0)
    total_time = _fmt_time(total_frames  / fps if fps else 0)
    cv2.putText(frame, cur_time, (pad, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 200), 1)
    tw, _ = cv2.getTextSize(total_time, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(frame, total_time, (w - pad - tw, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)


def _draw_corner_marks(frame, x1, y1, x2, y2, color, length=12, thickness=3):
    for (px, py, sx, sy) in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + sx * length, py), color, thickness)
        cv2.line(frame, (px, py), (px, py + sy * length), color, thickness)


def _draw_hud(frame, summary, target_model, loaded_model):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    fps         = summary.get("fps", 0)
    cnt         = summary.get("current_count", 0)
    tot         = summary.get("total_tracked", 0)
    model_label = loaded_model.replace(".pt", "").upper()

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 128), 2)
    cv2.putText(frame, f"LIVE: {cnt}  TOTAL: {tot}", (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 220), 1)

    counts = summary.get("class_counts_now", {})
    x_pos  = 240
    for cls, n in list(counts.items())[:7]:
        text = f"{cls}:{n}"
        cv2.putText(frame, text, (x_pos, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 100), 1)
        x_pos += len(text) * 9 + 10

    cv2.putText(frame, model_label, (w - 140, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (100, 200, 255), 2)


# =============================================================================
#  Instantiate processor
# =============================================================================
processor = VideoProcessor()


# =============================================================================
#  MJPEG stream generator
# =============================================================================

def _generate_stream(getter_fn, fps_cap=30):
    interval = 1.0 / fps_cap
    while True:
        t0    = time.time()
        frame = getter_fn()
        if frame:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        elapsed = time.time() - t0
        time.sleep(max(0, interval - elapsed))


# =============================================================================
#  Flask Routes
# =============================================================================

@app.route("/")
def index():
    return render_template(
        "index.html",
        models   = config.YOLO_MODELS,
        classes  = config.COCO_CLASSES,
        defaults = {
            "model"     : config.DEFAULT_MODEL,
            "confidence": config.DEFAULT_CONFIDENCE,
            "source"    : config.DEFAULT_SOURCE,
        }
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_stream(processor.get_frame),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/heatmap_feed")
def heatmap_feed():
    return Response(
        _generate_stream(processor.get_heatmap, fps_cap=5),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/analytics")
def api_analytics():
    return jsonify(processor.get_analytics())


# ── Playback controls ─────────────────────────────────────────────────────────

@app.route("/api/start", methods=["POST"])
def api_start():
    processor.start()
    return jsonify({"status": "ok", "state": processor.state})

@app.route("/api/pause", methods=["POST"])
def api_pause():
    processor.pause()
    return jsonify({"status": "ok", "state": processor.state})

@app.route("/api/resume", methods=["POST"])
def api_resume():
    processor.resume()
    return jsonify({"status": "ok", "state": processor.state})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    processor.stop()
    return jsonify({"status": "ok", "state": processor.state})

@app.route("/api/state")
def api_state():
    return jsonify({"state": processor.state})


# ── Seek / Scrub ──────────────────────────────────────────────────────────────

@app.route("/api/seek", methods=["POST"])
def api_seek():
    """
    Seek to a position in a video file.
    Body: { "position": 0.0–1.0 }   (fraction of total duration)
    e.g. { "position": 0.5 }  → jump to the middle of the video
    """
    data     = request.get_json(force=True)
    position = float(data.get("position", 0.0))
    processor.seek(position)
    return jsonify({"status": "ok", "position": position})


@app.route("/api/video_progress")
def api_video_progress():
    """
    Returns current playback position for the browser seek bar.
    Polled by the frontend every ~500ms while a video file is playing.
    """
    return jsonify(processor.get_video_progress())

# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(force=True)
    processor.apply_settings(data)
    return jsonify({"status": "ok"})

@app.route("/api/roi", methods=["GET"])
def api_roi_get():
    return jsonify(processor.roi.get_zones_json())

@app.route("/api/roi", methods=["POST"])
def api_roi_set():
    data  = request.get_json(force=True)
    zones = data.get("zones", [])
    processor.roi.set_zones(zones)
    return jsonify({"status": "ok", "count": len(zones)})

@app.route("/api/clear_heatmap", methods=["POST"])
def api_clear_heatmap():
    processor.heatmap.reset()
    return jsonify({"status": "ok"})

@app.route("/api/reset", methods=["POST"])
def api_reset():
    processor.tracker.reset()
    processor.analytics.reset()
    processor.heatmap.reset()
    return jsonify({"status": "ok"})

@app.route("/api/download_csv")
def api_download_csv():
    csv_path = os.path.join(config.LOG_DIR, config.CSV_FILENAME)
    if not os.path.exists(csv_path):
        return jsonify({"error": "No CSV log yet."}), 404
    return send_file(csv_path, as_attachment=True,
                     download_name="detections.csv", mimetype="text/csv")

@app.route("/api/upload_video", methods=["POST"])
def api_upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    save_dir  = config.LOG_DIR
    safe_name = os.path.basename(file.filename).replace(" ", "_")
    save_path = os.path.join(save_dir, safe_name)
    try:
        file.save(save_path)
        logger.info(f"Video uploaded: {save_path}")
        return jsonify({"status": "ok", "path": save_path})
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/models")
def api_models():
    return jsonify(list(config.YOLO_MODELS.keys()))

@app.route("/api/classes")
def api_classes():
    return jsonify(config.COCO_CLASSES)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"Starting server at http://localhost:{config.FLASK_PORT}")
    app.run(
        host    = config.FLASK_HOST,
        port    = config.FLASK_PORT,
        debug   = config.FLASK_DEBUG,
        threaded= True,
    )