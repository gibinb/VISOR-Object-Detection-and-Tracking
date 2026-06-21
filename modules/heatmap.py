# =============================================================================
#  modules/heatmap.py — Object Movement Heatmap Generator
# =============================================================================
#
#  Maintains a float32 accumulation map the same size as the video frame.
#  Each time an object centroid is detected, we add a Gaussian "blob" to
#  the map at that location.  The map is then normalised and colorised with
#  OpenCV's COLORMAP_JET (blue → green → yellow → red) and blended onto the
#  original frame.
#
#  Features
#  ─────────
#  • Per-frame decay so older visits fade over time
#  • Configurable radius, alpha blend, and decay rate
#  • Thread-safe reset
#  • Separate get_image() for the /heatmap_feed endpoint
#
# =============================================================================

import numpy as np
import cv2
import threading
import logging

from config import HEATMAP_ALPHA, HEATMAP_DECAY, HEATMAP_RADIUS

logger = logging.getLogger(__name__)


class HeatmapGenerator:
    """
    Accumulates centroid positions into a 2-D heat map.

    Usage
    -----
        hm = HeatmapGenerator()
        hm.set_size(frame_h, frame_w)
        hm.update(centroids)          # each frame
        blended = hm.blend(frame)     # overlay on frame
        img     = hm.get_image()      # standalone coloured heatmap JPEG
    """

    def __init__(self):
        self._lock  = threading.Lock()
        self._map   = None          # float32 accumulation array  (H, W)
        self._h     = 0
        self._w     = 0
        self._frame_count = 0

        # Pre-build a Gaussian kernel of the right radius
        self._kernel = self._make_gaussian(HEATMAP_RADIUS)

    # ── Setup ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_gaussian(radius: int) -> np.ndarray:
        """Build a 2-D Gaussian kernel (size = 2*radius+1)."""
        size = 2 * radius + 1
        k    = cv2.getGaussianKernel(size, radius / 3.0)
        kernel = k @ k.T                        # outer product → 2-D kernel
        return (kernel / kernel.max()).astype(np.float32)

    def set_size(self, height: int, width: int):
        """Initialise (or reinitialise) the accumulation map."""
        with self._lock:
            if self._h != height or self._w != width:
                self._map = np.zeros((height, width), dtype=np.float32)
                self._h   = height
                self._w   = width
                logger.info(f"Heatmap initialised: {width}×{height}")

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, centroids: list) -> None:
        """
        Add centroid blobs to the heatmap and apply temporal decay.

        Parameters
        ----------
        centroids : list of (cx, cy) tuples
        """
        if self._map is None:
            return

        with self._lock:
            # Temporal decay — older detections fade gradually
            self._map *= HEATMAP_DECAY

            r  = HEATMAP_RADIUS
            kh, kw = self._kernel.shape

            for cx, cy in centroids:
                cx, cy = int(cx), int(cy)

                # Compute the valid slice of the accumulation map
                y_start = max(0, cy - r)
                y_end   = min(self._h, cy + r + 1)
                x_start = max(0, cx - r)
                x_end   = min(self._w, cx + r + 1)

                # Corresponding slice in the kernel
                ky_start = y_start - (cy - r)
                ky_end   = ky_start + (y_end - y_start)
                kx_start = x_start - (cx - r)
                kx_end   = kx_start + (x_end - x_start)

                if (y_end > y_start and x_end > x_start and
                        ky_end > ky_start and kx_end > kx_start):
                    self._map[y_start:y_end, x_start:x_end] += \
                        self._kernel[ky_start:ky_end, kx_start:kx_end]

            self._frame_count += 1

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _colorise(self) -> np.ndarray:
        """Return a colourised BGR heatmap image (0-255 uint8)."""
        if self._map is None:
            return None
        max_val = self._map.max()
        if max_val < 1e-6:
            return np.zeros((self._h, self._w, 3), dtype=np.uint8)

        normalised  = (self._map / max_val * 255).astype(np.uint8)
        colourised  = cv2.applyColorMap(normalised, cv2.COLORMAP_JET)

        # Make zero-heat areas fully transparent (black → blend shows original)
        mask                   = normalised == 0
        colourised[mask]       = 0
        return colourised

    def blend(self, frame: np.ndarray) -> np.ndarray:
        """
        Overlay the heatmap on the given BGR frame.

        Returns a new blended frame (does not modify the original).
        """
        if self._map is None:
            return frame.copy()

        with self._lock:
            colourised = self._colorise()

        if colourised is None:
            return frame.copy()

        # Resize heatmap to frame dimensions if they differ
        fh, fw = frame.shape[:2]
        if colourised.shape[:2] != (fh, fw):
            colourised = cv2.resize(colourised, (fw, fh))

        # Create mask: only blend where heat > 0
        gray = cv2.cvtColor(colourised, cv2.COLOR_BGR2GRAY)
        mask = (gray > 0).astype(np.float32)[:, :, np.newaxis]

        blended = (frame.astype(np.float32) * (1 - mask * HEATMAP_ALPHA)
                   + colourised.astype(np.float32) * mask * HEATMAP_ALPHA)
        return blended.clip(0, 255).astype(np.uint8)

    def get_image(self) -> np.ndarray:
        """
        Return a standalone colourised heatmap image (for the heatmap endpoint).
        Same size as the video frame, with a black background where heat = 0.
        """
        with self._lock:
            colourised = self._colorise()
        if colourised is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Add a subtle legend bar on the right
        out = colourised.copy()
        legend_w = 20
        if out.shape[1] > legend_w + 10:
            bar = np.linspace(255, 0, out.shape[0], dtype=np.uint8)
            bar_img = np.tile(bar[:, np.newaxis], (1, legend_w))
            bar_col = cv2.applyColorMap(bar_img, cv2.COLORMAP_JET)
            out[:, -legend_w:] = bar_col
        return out

    # ── Control ───────────────────────────────────────────────────────────────

    def reset(self):
        """Clear the heatmap accumulation."""
        with self._lock:
            if self._map is not None:
                self._map[:] = 0
            self._frame_count = 0
        logger.info("Heatmap reset.")

    @property
    def frame_count(self) -> int:
        return self._frame_count
