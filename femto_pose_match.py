"""Real-time 3D pose, distance and velocity for one person from an
Orbbec Femto Bolt.

Pose is detected on the active-IR image (works in the dark / under stage
lights). IR and depth come from the same ToF sensor, so they are pixel-aligned:
each 2D keypoint gets a real depth reading, back-projected to metres (maths
stays in metres; everything shown is in feet).

Output is two torso numbers (shoulder-hip region, depth averaged over the
whole torso, smoothed by the same filter that gives the speed):
    depth (ft) and speed (ft/s)
sent every frame as one OSC message to 127.0.0.1:7000 (--osc-host / --osc-port
to change, --no-osc to turn off): address /femto, one string argument holding a
pretty-printed JSON object. Both values read "N/A" when no one is there:
    {
      "depth_ft": 6.42,
      "speed_ft_s": 0.31,
      "arm_extension": 0.42
    }
"arm_extension" is 0 with hands together and 1 with arms straight out to the
sides (see arm_extension()); "N/A" when an arm can't be seen.

The RGB colour camera is also sent out as a virtual webcam called "Imagineers"
(--vcam-name to change, --no-vcam to turn off): pick it like any camera in
TouchDesigner (Video Device In), OBS, Zoom etc. It uses the Unity Capture
DirectShow driver, registered under that name by install_virtualcam.bat
(run_pose_match.bat does this on first run).

Low latency: the camera runs in callback mode and colour is not held back to
sync with depth. The virtual camera and pose each run on their own thread and always take the
newest frame (stale frames are dropped, never queued), so delay can't build up.

This script shows a preview window.

Keys:  q / Esc = quit   d = toggle depth panel   r = reset the tracker
       p = snapshot to ./snapshots (preview, IR, colour + JSON of the arm landmarks)
"""

import argparse
import json
import socket
import struct
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from pyorbbecsdk import Config, OBFormat, OBFrameAggregateOutputMode, OBSensorType, Pipeline

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
TORSO = (11, 12, 24, 23)  # L/R shoulder, R/L hip (polygon order): stable anchor for distance / speed
M_TO_FT = 3.28084  # display unit: feet
VIS_THRESH = 0.5
PATCH = 3  # half-size of depth sampling patch (7x7)
TORSO_SHRINK = 0.6  # sample torso depth inside the shoulder-hip quad scaled to 60% (stays off the edges)
TORSO_BAND = 0.25  # m: torso pixels further than this from the median are background seen through gaps
KP_JITTER_PX = 2.0  # typical pose-keypoint jitter (px) -> lateral measurement noise grows with distance
TORSO_Z_STD = 0.02  # m: torso-region depth noise incl. body sway; erring high is cheap, too low lets noise into speed
COLOUR = (255, 220, 0)  # BGR: cyan skeleton / text
OSC_ADDRESS = "/femto"  # one OSC address carrying the JSON object
PAIR_TOL_US = 16000  # depth and IR frames closer than this in time belong to the same capture


def load_model(variant: str) -> Path:
    path = Path(__file__).with_name(f"pose_landmarker_{variant}.task")
    if not path.exists():
        print(f"Downloading {variant} pose model…")
        urllib.request.urlretrieve(MODEL_URLS[variant], path)
    return path


class Latest:
    """One-slot mailbox: put() overwrites, get() takes the newest item (or None on timeout).
    A slow consumer skips frames instead of falling behind."""

    def __init__(self):
        self._item = None
        self._cv = threading.Condition()

    def put(self, item):
        with self._cv:
            self._item = item
            self._cv.notify()

    def get(self, timeout):
        with self._cv:
            if self._item is None:
                self._cv.wait(timeout)
            item, self._item = self._item, None
            return item


def start_camera(width: int, color_res, pose_slot: Latest, color_slot):
    """Start pixel-aligned depth + IR streams, plus MJPG colour if color_res=(w, h).
    (MJPG and decode it ourselves: asking the SDK for BGRA uses Windows' hardware
    decoder, which can fail with "lack of hardware resources".)

    Runs in callback mode: the SDK hands us each frame as soon as it arrives.
    Colour goes straight to color_slot (not held back to sync with depth); depth + IR
    from the same capture are paired and put in pose_slot as (depth_raw, scale, ir, ts_us).
    Returns (pipeline, intrinsics)."""
    pipeline = Pipeline()
    config = Config()
    if color_res:
        profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        profile = profiles.get_video_stream_profile(color_res[0], color_res[1], OBFormat.MJPG, 30)
        config.enable_stream(profile)
        print(f"Colour: {profile.get_width()}x{profile.get_height()} MJPG @ {profile.get_fps()} fps")
    for sensor, fmt in ((OBSensorType.DEPTH_SENSOR, OBFormat.Y16), (OBSensorType.IR_SENSOR, OBFormat.Y16)):
        profiles = pipeline.get_stream_profile_list(sensor)
        try:
            profile = profiles.get_video_stream_profile(width, 0, fmt, 30)
        except Exception:
            profile = profiles.get_default_video_stream_profile()
        config.enable_stream(profile)
        if sensor == OBSensorType.DEPTH_SENSOR:
            intr = profile.get_intrinsic()
            print(f"Depth: {profile.get_width()}x{profile.get_height()} @ {profile.get_fps()} fps")
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.ANY_SITUATION)  # deliver whatever is ready
    pipeline.disable_frame_sync()  # don't delay colour to line up with depth

    pending = {}  # newest unpaired depth / IR

    def on_frames(fs):
        if fs is None:
            return
        cf = fs.get_color_frame()
        if cf is not None and color_slot is not None:
            color_slot.put(np.frombuffer(cf.get_data(), np.uint8).copy())
        df, irf = fs.get_depth_frame(), fs.get_ir_frame()
        if df is not None:
            raw = np.frombuffer(df.get_data(), np.uint16).reshape(df.get_height(), df.get_width()).copy()
            pending["depth"] = (df.get_timestamp_us(), raw, df.get_depth_scale())
        if irf is not None:
            raw = np.frombuffer(irf.get_data(), np.uint16).reshape(irf.get_height(), irf.get_width()).copy()
            pending["ir"] = (irf.get_timestamp_us(), raw)
        if "depth" in pending and "ir" in pending:
            (dts, depth, scale), (its, ir) = pending["depth"], pending["ir"]
            if abs(dts - its) <= PAIR_TOL_US:
                pose_slot.put((depth, scale, ir, dts))
                pending.clear()

    pipeline.start(config, on_frames)
    return pipeline, intr


def start_vcam(name: str, w: int, h: int):
    """Virtual webcam taking BGR frames, via the Unity Capture driver registered as `name`."""
    import pyvirtualcam
    try:
        cam = pyvirtualcam.Camera(w, h, 30, fmt=pyvirtualcam.PixelFormat.BGR, backend="unitycapture", device=name)
    except RuntimeError as e:
        sys.exit(f'Could not open virtual camera "{name}": {e}\nRun install_virtualcam.bat (it asks for admin) '
                 "to register it, or start with --no-vcam.")
    print(f'Virtual camera: sending colour as "{cam.device}" ({w}x{h})')
    return cam


def vcam_loop(cam, color_slot: Latest, color_res, stop: threading.Event, latest_colour: dict):
    """Decode + send each colour frame the moment it arrives, independent of pose speed.
    The newest decoded frame is also kept in latest_colour["bgr"] for snapshots."""
    while not stop.is_set():
        jpg = color_slot.get(0.2)
        if jpg is None:
            continue
        bgr = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != color_res[::-1]:
            continue
        latest_colour["bgr"] = bgr
        cam.send(bgr)


def osc_message(address: str, value) -> bytes:
    """Minimal OSC 1.0 message with one string, int or float argument."""
    def pad(b):
        return b + b"\0" * (4 - len(b) % 4)
    if isinstance(value, str):
        tag, arg = b",s", pad(value.encode())
    elif isinstance(value, int):
        tag, arg = b",i", struct.pack(">i", value)
    else:
        tag, arg = b",f", struct.pack(">f", value)
    return pad(address.encode()) + pad(tag) + arg


def ir_to_rgb(ir: np.ndarray) -> np.ndarray:
    """16-bit IR -> contrast-stretched 8-bit RGB for the pose model."""
    lo, hi = np.percentile(ir[ir > 0], (1, 99)) if np.any(ir > 0) else (0, 1)
    img = np.clip((ir.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def sample_depth_m(depth_m: np.ndarray, u: int, v: int) -> float:
    """Median of valid depth in a small patch around (u, v); NaN if none."""
    h, w = depth_m.shape
    patch = depth_m[max(v - PATCH, 0):min(v + PATCH + 1, h), max(u - PATCH, 0):min(u + PATCH + 1, w)]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else float("nan")


def back_project(u, v, z, intr):
    return ((u - intr.cx) * z / intr.fx, (v - intr.cy) * z / intr.fy, z)


class VelocityIMM:
    """Velocity tracker for a 3D point (metres, seconds): an interacting-multiple-model
    (IMM) filter blending two constant-velocity Kalman filters.

    A single filter must trade noise for lag: smooth enough to read ~0 when someone
    stands still makes it slow to pick up a walk-off, and vice versa. Here a "calm"
    model (barely any acceleration) and a "manoeuvre" model (quick changes) run side
    by side, and each frame they are weighted by how well they explain the data, so
    the reading is quiet while still or steady and still reacts within ~2 frames.
    Simulated with realistic Femto noise vs the old acceleration filter: still-speed
    noise 0.46 -> ~0.15 ft/s, error while moving 1.1 -> ~0.5 ft/s.

    Measurements that jump implausibly far (e.g. the depth hitting the wall behind)
    are rejected statistically; the track coasts through missed frames.
    """

    def __init__(self, calm_acc=0.3, manoeuvre_acc=6.0, p_stay=0.97, n_sigma=4.0, gate_floor=0.05,
                 reanchor=0.25, max_coast=1.0):
        self.q = np.array([calm_acc, manoeuvre_acc]) ** 2  # white-noise acceleration (m/s^2)^2 per model
        self.trans = np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])  # model switching odds per frame
        self.n_sigma, self.gate_floor = n_sigma, gate_floor
        self.reanchor, self.max_coast = reanchor, max_coast
        self.reset()

    def reset(self):
        self.x = None  # (model, axis, [p, v])
        self.coast = 0.0  # s since the last accepted measurement
        self.rejected = 0.0  # s of back-to-back gate rejections

    def _init(self, meas, r):
        self.x = np.zeros((2, 3, 2))
        self.x[:, :, 0] = meas
        self.P = np.zeros((2, 3, 2, 2))
        self.P[:, :, 0, 0] = r
        self.P[:, :, 1, 1] = 1.0  # velocity unknown at start (~1 m/s std)
        self.mu = np.array([0.5, 0.5])
        self.coast = self.rejected = 0.0

    @property
    def tracking(self):
        return self.x is not None

    @property
    def pos(self):
        return self.mu @ self.x[:, :, 0]

    @property
    def vel(self):
        return self.mu @ self.x[:, :, 1]

    def predict_pos(self, dt):
        return self.pos + self.vel * dt

    def update(self, meas, dt, meas_std):
        """meas: (3,) position or None; meas_std: (3,) its noise (m). Returns True while tracking."""
        r = np.asarray(meas_std, float) ** 2
        if self.x is None:
            if meas is not None:
                self._init(meas, r)
            return self.tracking
        # 1. mix: each model restarts from a blend of both, weighted by switching odds
        c = self.trans.T @ self.mu
        w = self.trans * self.mu[:, None] / c[None, :]  # w[i, j] = P(was model i | now model j)
        x0 = np.einsum("ij,iak->jak", w, self.x)
        d = self.x[:, None] - x0[None]
        P0 = np.einsum("ij,ijakl->jakl", w, self.P[:, None] + d[..., :, None] * d[..., None, :])
        # 2. predict (constant velocity, white-noise acceleration)
        F = np.array([[1, dt], [0, 1]])
        Qb = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        xp = x0 @ F.T
        Pp = F @ P0 @ F.T + self.q[:, None, None, None] * Qb
        self.x, self.P, self.mu = xp, Pp, c
        self.coast += dt
        if self.coast > self.max_coast:
            self.reset()
            return False
        if meas is None:
            return True
        # 3. gate on the blended prediction
        meas = np.asarray(meas, float)
        S = Pp[:, :, 0, 0] + r  # (model, axis)
        if np.any(np.abs(meas - self.pos) > self.n_sigma * np.sqrt(c @ S) + self.gate_floor):
            self.rejected += dt
            if self.rejected > self.reanchor:  # consistently elsewhere: the person really is there
                self._init(meas, r)
            return True
        # 4. update each model, then reweight models by how well they predicted this measurement
        innov = meas[None] - xp[:, :, 0]
        K = Pp[:, :, :, 0] / S[..., None]
        self.x = xp + K * innov[..., None]
        self.P = Pp - K[..., :, None] * Pp[:, :, 0, None, :]
        like = np.exp(-0.5 * np.sum(innov ** 2 / S, axis=1)) / np.sqrt(np.prod(S, axis=1))
        mu = c * like + 1e-300
        self.mu = mu / mu.sum()
        self.coast = self.rejected = 0.0
        return True


def torso_depth_m(depth_m, quad):
    """Median depth over the torso itself (the shoulder-hip quad shrunk toward its
    centre), not a few 7x7 patches on the joints: thousands of pixels instead of ~200,
    and none on the body's edge where a patch can slip onto the wall behind.
    Pixels far from the median (background seen past an arm / gap) are dropped."""
    centre = quad.mean(axis=0)
    poly = np.round(centre + (quad - centre) * TORSO_SHRINK).astype(np.int32)
    h, w = depth_m.shape
    x0, y0 = np.clip(poly.min(axis=0), 0, [w - 1, h - 1])
    x1, y1 = np.clip(poly.max(axis=0) + 1, 1, [w, h])
    if x1 - x0 < 3 or y1 - y0 < 3:
        return float("nan")
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(mask, poly - [x0, y0], 1)
    vals = depth_m[y0:y1, x0:x1][(mask > 0) & (depth_m[y0:y1, x0:x1] > 0)]
    if vals.size < 20:
        return float("nan")
    med = np.median(vals)
    vals = vals[np.abs(vals - med) < TORSO_BAND]
    return float(np.median(vals))


def detection_centre(px, z, depth_m, intr):
    """Torso centre: 2D centroid of shoulders+hips (always present, even when low
    visibility) at the torso-region depth -> stable when single joints flicker.
    Falls back to the median of the joint depths if the torso region has no depth."""
    quad = np.array([px[i] for i in TORSO], float)
    d = torso_depth_m(depth_m, quad)
    if not np.isfinite(d):
        torso = [z[i] for i in TORSO if np.isfinite(z[i])]
        if not torso:
            return None, None
        d = float(np.median(torso))
    u, v = quad.mean(axis=0)
    return np.array(back_project(u, v, d, intr)), d


# --- arm extension ----------------------------------------------------------------------------
ARM_JOINTS = (11, 12, 13, 14, 15, 16)  # L/R shoulder, elbow, wrist
ARM_VIS = 0.3  # below this an arm joint is too unsure to use
ARM_SMOOTH = 0.35  # EMA weight of the newest frame (lower = smoother, laggier)
# Calibration from snapshots on the Femto (2026-11-17, full model): hands clasped in front
# read a raw 0.148 / 0.157 / 0.175 (the model puts the wrists ~20 cm apart at the edges of
# the clasped hands), arms straight out read 0.943. The raw ratio is mapped so that
# ARM_ZERO -> 0 and ARM_FULL -> 1 (--arm-zero / --arm-full to change).
ARM_ZERO = 0.17
ARM_FULL = 0.92


def arm_ratio(world, lms):
    """Raw ratio: wrist gap / (shoulder width + both upper arms + both forearms).

    With arms straight out the wrists are as far apart as the whole chain is long, so
    the ratio is ~1; with hands together it is small but not 0 (see ARM_ZERO). Uses
    MediaPipe's world landmarks (metres, centred on the hips), so it doesn't depend
    on how far the person is from the camera, and being a ratio of the person's own
    lengths it works for any body size. The arm lengths are full 3D (an arm pointing
    at the camera keeps its true length); the wrist gap is measured across / up-down
    only, because the model's front-back guess for hands held in front of the body is
    noisy (up to ~20 cm between two clasped wrists in testing).
    Returns None if an arm joint isn't visible enough."""
    if world is None or any(lms[i].visibility < ARM_VIS for i in ARM_JOINTS):
        return None
    p = {i: np.array([world[i].x, world[i].y, world[i].z]) for i in ARM_JOINTS}
    dist = lambda a, b: float(np.linalg.norm(p[a] - p[b]))
    chain = dist(11, 12) + dist(11, 13) + dist(13, 15) + dist(12, 14) + dist(14, 16)
    if chain < 1e-6:
        return None
    gap = float(np.linalg.norm(p[15][:2] - p[16][:2]))
    return gap / chain


def arm_extension(world, lms, zero=ARM_ZERO, full=ARM_FULL):
    """0 = hands together, 1 = arms straight out to the sides (full wingspan), clipped to
    0..1: arm_ratio() rescaled so `zero` maps to 0 and `full` to 1. None if arms hidden."""
    r = arm_ratio(world, lms)
    if r is None:
        return None
    return float(np.clip((r - zero) / max(full - zero, 1e-6), 0.0, 1.0))


LANDMARK_NAMES = {0: "nose", 11: "l_shoulder", 12: "r_shoulder", 13: "l_elbow", 14: "r_elbow", 15: "l_wrist",
                  16: "r_wrist", 17: "l_pinky", 18: "r_pinky", 19: "l_index", 20: "r_index", 21: "l_thumb",
                  22: "r_thumb", 23: "l_hip", 24: "r_hip"}


def arm_debug(world, lms):
    """Everything behind the arm-extension number, for a snapshot (p key)."""
    if world is None or lms is None:
        return {"person": False}
    p = lambda i: np.array([world[i].x, world[i].y, world[i].z])
    dist = lambda a, b: round(float(np.linalg.norm(p(a) - p(b))), 4)
    segs = {"shoulder_width": dist(11, 12), "l_upper_arm": dist(11, 13), "l_forearm": dist(13, 15),
            "r_upper_arm": dist(12, 14), "r_forearm": dist(14, 16)}
    return {
        "person": True,
        "arm_vis_threshold": ARM_VIS,
        "joints": {name: {"world_m": [round(world[i].x, 4), round(world[i].y, 4), round(world[i].z, 4)],
                          "image_xy": [round(lms[i].x, 4), round(lms[i].y, 4)],
                          "visibility": round(lms[i].visibility, 3)} for i, name in LANDMARK_NAMES.items()},
        "segments_m": segs,
        "arm_span_m": round(sum(segs.values()), 4),
        "wrist_distance_m": dist(15, 16),
        "index_tip_distance_m": dist(19, 20),
        "wrist_gap_xy_m": round(float(np.hypot(world[15].x - world[16].x, world[15].y - world[16].y)), 4),
        "raw_arm_ratio": arm_ratio(world, lms),  # before the zero / full rescale
    }


def parse_args(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model", choices=MODEL_URLS, default="full", help="pose model: lite (fastest), full (default, better hands), heavy (most accurate, slowest)")
    ap.add_argument("--width", type=int, default=640, help="640 = NFOV unbinned (1.6–12.7 ft), 320 = NFOV binned (to ~18 ft)")
    ap.add_argument("--vcam-name", default="Imagineers", help="virtual camera name for the colour camera")
    ap.add_argument("--no-vcam", action="store_true", help="don't send the colour camera to the virtual camera")
    ap.add_argument("--osc-host", default="127.0.0.1", help="where to send the OSC data")
    ap.add_argument("--osc-port", type=int, default=7000, help="OSC port (TouchDesigner OSC In DAT)")
    ap.add_argument("--no-osc", action="store_true", help="don't send OSC data")
    ap.add_argument("--color-res", default="1920x1080", help="colour resolution for the virtual camera, e.g. 1280x720, 1920x1080, 3840x2160")
    ap.add_argument("--arm-zero", type=float, default=ARM_ZERO, help="raw arm ratio that counts as 0 (hands together); see raw_arm_ratio in a p snapshot")
    ap.add_argument("--arm-full", type=float, default=ARM_FULL, help="raw arm ratio that counts as 1 (arms straight out)")
    return ap.parse_args()


def run(args, on_frame):
    """Camera -> pose -> torso filter -> OSC, once per depth/IR frame.
    on_frame(rgb, depth_m, det, depth_ft, speed_ft_s, fps, torso, arm_ext, info) is called after each
    frame for drawing / status (rgb is None if no frame came within 0.5 s); info is a dict with
    "world", "lms" (this frame's landmarks, or None) and "colour" (newest colour frame, or None),
    for snapshots. Return False from it to stop."""
    color_res = None if args.no_vcam else tuple(int(n) for n in args.color_res.lower().split("x"))

    landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(load_model(args.model))),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    ))

    pose_slot = Latest()
    color_slot = Latest() if color_res else None
    try:
        pipeline, intr = start_camera(args.width, color_res, pose_slot, color_slot)
    except RuntimeError as e:
        sys.exit(f"Could not open Femto Bolt: {e}\nCheck it is on a USB 3 port with its power supply connected, and that "
                 "Orbbec Viewer / other apps are closed. (Femto Bolt is not supported on macOS.)")

    stop = threading.Event()
    latest_colour = {"bgr": None}  # newest colour frame, for snapshots
    vcam = start_vcam(args.vcam_name, *color_res) if color_res else None
    vcam_thread = None
    if vcam is not None:
        vcam_thread = threading.Thread(target=vcam_loop, args=(vcam, color_slot, color_res, stop, latest_colour),
                                       daemon=True)
        vcam_thread.start()
    osc = None if args.no_osc else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if osc is not None:
        print(f"OSC: sending JSON on {OSC_ADDRESS} to {args.osc_host}:{args.osc_port}")

    t0, last = time.monotonic(), time.monotonic()
    fps = 0.0
    torso = VelocityIMM(max_coast=1.0)  # rides out brief occlusion
    prev_ts_us = None
    arm_ext = None  # smoothed arm extension 0..1
    try:
        while True:
            item = pose_slot.get(0.5)
            if item is None:
                if on_frame(None, None, None, None, None, fps, torso, None, None) is False:
                    break
                continue
            depth_raw, scale, ir, ts_us = item
            h, w = depth_raw.shape
            depth_m = depth_raw * (scale / 1000.0)

            # dt from the camera's own clock, so slow processing / skipped frames don't distort speed
            dt = (ts_us - prev_ts_us) / 1e6 if prev_ts_us and ts_us > prev_ts_us else 1 / 30
            prev_ts_us = ts_us

            rgb = ir_to_rgb(ir)
            ts = int((time.monotonic() - t0) * 1000)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            # --- 2D + depth for the detected person -----------------------------------------
            det, c, d, lms = None, None, None, None
            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                px = [(int(l.x * w), int(l.y * h)) for l in lms]
                ok = [l.visibility > VIS_THRESH and 0 <= x < w and 0 <= y < h for l, (x, y) in zip(lms, px)]
                z = [sample_depth_m(depth_m, x, y) if o else float("nan") for (x, y), o in zip(px, ok)]
                c, d = detection_centre(px, z, depth_m, intr)
                det = (px, ok, z)

            # --- update filter --------------------------------------------------------------
            # keypoint jitter is fixed in pixels, so in metres it grows with distance
            lateral = KP_JITTER_PX * (d or 3.0) / np.array([intr.fx, intr.fy])
            torso.update(c, dt, [lateral[0], lateral[1], TORSO_Z_STD])
            # torso average: filtered torso depth and the speed derived from the same filter
            depth_ft = float(torso.pos[2] * M_TO_FT) if torso.tracking else None
            speed_ft_s = float(np.linalg.norm(torso.vel) * M_TO_FT) if torso.tracking else None

            # --- arm extension (0 = hands together, 1 = full wingspan), lightly smoothed ----
            world = result.pose_world_landmarks[0] if lms is not None and result.pose_world_landmarks else None
            raw_arm = arm_extension(world, lms, args.arm_zero, args.arm_full) if world is not None else None
            if raw_arm is None:
                arm_ext = None
            else:
                arm_ext = raw_arm if arm_ext is None else arm_ext + ARM_SMOOTH * (raw_arm - arm_ext)

            # --- data out (OSC) -------------------------------------------------------------
            if osc is not None:
                data = {
                    "depth_ft": round(depth_ft, 2) if torso.tracking else "N/A",
                    "speed_ft_s": round(speed_ft_s, 2) if torso.tracking else "N/A",
                    "arm_extension": round(arm_ext, 3) if arm_ext is not None else "N/A",
                }
                try:
                    osc.sendto(osc_message(OSC_ADDRESS, json.dumps(data, indent=2)), (args.osc_host, args.osc_port))
                except OSError:
                    pass  # receiver not up yet; keep running

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            info = {"world": world, "lms": lms, "colour": latest_colour["bgr"]}
            if on_frame(rgb, depth_m, det, depth_ft, speed_ft_s, fps, torso, arm_ext, info) is False:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        pipeline.stop()
        if vcam_thread is not None:
            vcam_thread.join(1.0)
        if vcam is not None:
            vcam.close()
        if osc is not None:
            osc.close()
        landmarker.close()


def main():
    args = parse_args("Femto Bolt pose -> OSC depth/speed + virtual camera colour, with preview window")
    font = cv2.FONT_HERSHEY_SIMPLEX
    state = {"show_depth": True}

    def snapshot(rgb, view, arm_ext, info):
        """p key: save what the camera / model sees right now to ./snapshots."""
        folder = Path(__file__).with_name("snapshots")
        folder.mkdir(exist_ok=True)
        stem = folder / time.strftime("snap_%Y%m%d_%H%M%S")
        cv2.imwrite(f"{stem}_preview.png", view)  # IR with skeleton + bars (+ depth panel if shown)
        cv2.imwrite(f"{stem}_ir.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))  # exactly what the model got
        if info["colour"] is not None:
            cv2.imwrite(f"{stem}_colour.jpg", info["colour"])
        data = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": args.model,
                "arm_zero": args.arm_zero, "arm_full": args.arm_full,
                "arm_extension_smoothed": arm_ext, **arm_debug(info["world"], info["lms"])}
        Path(f"{stem}.json").write_text(json.dumps(data, indent=2))
        print(f"Snapshot saved: {stem.name}_*  (arm extension {arm_ext if arm_ext is None else round(arm_ext, 3)})")

    def draw(rgb, depth_m, det, depth_ft, speed_ft_s, fps, torso, arm_ext, info):
        if rgb is None:  # no frame this time; keep the window responsive
            return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)
        h, w = depth_m.shape
        view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        skel = COLOUR
        if det is not None:
            px, ok, z = det
            for a, b in CONNECTIONS:
                if ok[a] and ok[b]:
                    cv2.line(view, px[a], px[b], skel, 2)
            for i, ((x, y), o) in enumerate(zip(px, ok)):
                if o:
                    cv2.circle(view, (x, y), 4, skel if np.isfinite(z[i]) else (0, 0, 255), -1)

        header = 34
        cv2.rectangle(view, (0, 0), (w, header), (0, 0, 0), -1)
        line = f"Depth {depth_ft:.2f} ft   Speed {speed_ft_s:.2f} ft/s" if torso.tracking else "No one detected"
        cv2.putText(view, line, (10, 22), font, 0.55, COLOUR, 2, cv2.LINE_AA)
        cv2.putText(view, f"{fps:4.1f} fps", (w - 75, header - 6), font, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

        # arm extension bar along the bottom: 0 = hands together, 1 = full wingspan
        cv2.rectangle(view, (0, h - 30), (w, h), (0, 0, 0), -1)
        label = f"Arms {arm_ext:.2f}" if arm_ext is not None else "Arms N/A"
        cv2.putText(view, label, (10, h - 10), font, 0.55, COLOUR, 2, cv2.LINE_AA)
        bx0, bx1 = 120, w - 10
        cv2.rectangle(view, (bx0, h - 22), (bx1, h - 10), (60, 60, 60), -1)
        if arm_ext is not None:
            cv2.rectangle(view, (bx0, h - 22), (bx0 + int((bx1 - bx0) * arm_ext), h - 10), COLOUR, -1)

        if state["show_depth"]:
            dvis = cv2.applyColorMap(cv2.convertScaleAbs(depth_m, alpha=255 / 5.0), cv2.COLORMAP_TURBO)
            dvis[depth_m == 0] = 0
            view = np.hstack([view, cv2.resize(dvis, (view.shape[1], view.shape[0]))])
        cv2.imshow("Femto Bolt pose (IR + depth)", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key == ord("d"):
            state["show_depth"] = not state["show_depth"]
        if key == ord("r"):  # reset the tracker
            torso.reset()
        if key == ord("p"):  # snapshot for checking the arm-extension mapping
            snapshot(rgb, view, arm_ext, info)
        return True

    try:
        run(args, draw)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
