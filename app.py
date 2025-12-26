#!/usr/bin/env python3
import os
import time
import json
import glob
import queue
import psutil
import pytz
import torch
import cv2
import numpy as np
import threading
import requests

from ultralytics import YOLO
from collections import defaultdict
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta


# ==========================
# Environment-driven config
# ==========================
# Required/expected env vars for Docker deployments:
#   CAMERA_ID           (default: Camera_1)
#   OUTPUT_VIDEO_LENGTH (minutes, default: 15)
#   BASE_DIR            (default: /data/output/<CAMERA_ID>)
#   CONFIG_JSON         (default: /app/config/Camera_Calibration_Info.json)
#   DAY_MODEL           (default: /app/models/bestdaytimeOcr3.pt)
#   NIGHT_MODEL         (default: /app/models/best_night_caltrans_0718.pt)
#   TIMEZONE            (default: US/Pacific)
#   DB_URL              (default: https://a.com/c/ingest/upload)
#   RTSP_URL            (optional override; if set, overrides config rtsp_url)

CAMERA_ID = os.getenv("CAMERA_ID", "Camera_1")
OUTPUT_VIDEO_LENGTH = int(os.getenv("OUTPUT_VIDEO_LENGTH", "15"))

CONFIG_JSON = os.getenv("CONFIG_JSON", "/app/config/Camera_Calibration_Info.json")
DAY_MODEL_PATH = os.getenv("DAY_MODEL", "/app/models/bestdaytimeOcr3.pt")
NIGHT_MODEL_PATH = os.getenv("NIGHT_MODEL", "/app/models/best_night_caltrans_0718.pt")

TIMEZONE = os.getenv("TIMEZONE", "US/Pacific")
TZ = pytz.timezone(TIMEZONE)
# DB_URL = os.getenv("DB_URL", "https://aiwaysionapi.com/caltrans-davis/ingest/sent")
DB_URL = os.getenv("DB_URL", "https://aiwaysionapi.com/caltrans-davis/ingest/upload")

BASE_DIR = os.getenv("BASE_DIR", f"/data/output/{CAMERA_ID}")
os.makedirs(BASE_DIR, exist_ok=True)

process = psutil.Process(os.getpid())


# --------------------------
# Day/Night model selection
# --------------------------
def model_selection_criteria(models, timenow=None):
    """
    Select daytime vs nighttime model based on local time.
    daytime: 06:00–17:59
    nighttime: 18:00–05:59
    """
    if timenow is None:
        timenow = datetime.now(TZ)

    nighttime_threshold = 18  # after 18:00 -> nighttime model
    daytime_threshold = 6     # from 06:00 -> daytime model

    current_hour = timenow.hour
    if daytime_threshold <= current_hour < nighttime_threshold:
        return models["daytime"], "daytime"
    else:
        return models["nighttime"], "nighttime"


# ==========================
# Utility Functions
# ==========================
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    return obj


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def lines_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def haversine(lat1, lon1, lat2, lon2):
    """Compute distance (km) between two geographic points."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def floor_to_5min(dt: datetime) -> datetime:
    minute = (dt.minute // 5) * 5
    return dt.replace(minute=minute, second=0, microsecond=0)


def get_output_video_path(base_dir, cam_num, output_video_length_min):
    now = datetime.now(TZ)
    clip_start = now.replace(
        minute=(now.minute // output_video_length_min) * output_video_length_min,
        second=0,
        microsecond=0,
    )
    suffix = clip_start.strftime("%Y-%m-%d_%H-%M")
    return os.path.join(base_dir, f"output_cam_{cam_num}_{suffix}.mp4"), clip_start


def delete_previous_day_clips(base_dir, cam_num):
    yesterday = (datetime.now(TZ).date() - timedelta(days=1))
    pattern = os.path.join(base_dir, f"output_cam_{cam_num}_{yesterday}*.mp4")
    for filepath in glob.glob(pattern):
        try:
            os.remove(filepath)
            print(f"Deleted old clip: {filepath}")
        except Exception as e:
            print(f"Failed to delete {filepath}: {e}")


def send_five_min_log(json_path: str, url: str = DB_URL):
    """Read the 5-min JSON file, strip 'ids', and POST to the DB."""
    if not json_path or not os.path.exists(json_path):
        return

    try:
        with open(json_path, "r") as f:
            entries = json.load(f)
    except Exception:
        return

    if not entries:
        return

    payload = []
    for e in entries:
        d = dict(e)
        d.pop("ids", None)
        payload.append(d)

    try:
        requests.post(url, json={"data": payload}, timeout=10)
    except Exception:
        pass


def flush_five_min_window(start_dt: datetime, end_dt: datetime,
                          countline_ids, turning_ids,
                          out_entries: list):
    """Append 5-min aggregates to out_entries in the requested JSON format."""
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    # Turning movements: key = (zone_pair, mode_id)
    for (zone_pair, mode_id), id_set in turning_ids.items():
        if not id_set or int(mode_id) in (1, 2, 7):  # skip pedestrian/wheelchair/animals
            continue
        out_entries.append({
            "ids": sorted(int(tid) for tid in id_set),
            "count": len(id_set),
            "start_datetime": start_str,
            "end_datetime": end_str,
            "turn_id": zone_pair,
            "mode": int(mode_id),
        })

    # Countlines: key = (line_id, dir_label, mode_id)
    for (line_id, dir_label, mode_id), id_set in countline_ids.items():
        if not id_set:
            continue
        out_entries.append({
            "ids": sorted(int(tid) for tid in id_set),
            "count": len(id_set),
            "start_datetime": start_str,
            "end_datetime": end_str,
            "countline_id": str(line_id),
            "countline_dir": dir_label,
            "mode": int(mode_id),
        })


# ==========================
# Load config
# ==========================
with open(CONFIG_JSON, "r") as f:
    config = json.load(f)

if CAMERA_ID not in config:
    raise KeyError(f"Camera ID '{CAMERA_ID}' not found in config {CONFIG_JSON}")

counting_lines = config[CAMERA_ID]["counting_lines"]

# Load input video path from config (or override via env)
input_video_path = os.getenv("RTSP_URL") or config[CAMERA_ID].get("rtsp_url")
if not input_video_path:
    raise ValueError(f"No rtsp_url found for {CAMERA_ID} in {CONFIG_JSON} and RTSP_URL not set.")

raw_zone_polygons = config[CAMERA_ID]["zone_polygons"]

# Pre-cast all zone polygons to float32 arrays
zone_polygons = {
    label: np.array(poly, dtype=np.float32)
    for label, poly in raw_zone_polygons.items()
}

# Prepare multiple homographies for each ROI
roi_homographies = []
roi_polygons_for_draw = []

for key in config[CAMERA_ID]:
    if key.startswith("ROI_") and key.endswith("_PIXEL"):
        roi_index = key.split("_")[1]
        pixel_key = f"ROI_{roi_index}_PIXEL"
        geo_key = f"ROI_{roi_index}_GEO"

        if geo_key in config[CAMERA_ID]:
            roi_pixels = np.array(config[CAMERA_ID][pixel_key], dtype=np.float32)
            roi_geo = np.array(config[CAMERA_ID][geo_key], dtype=np.float32)
            H, _ = cv2.findHomography(roi_pixels, roi_geo)

            roi_homographies.append({
                "pixel_polygon": roi_pixels,
                "homography": H
            })
            roi_polygons_for_draw.append(roi_pixels)

if not roi_homographies:
    print("[WARN] No ROI homographies were built. pixel_to_geo() will return None for all points.")


def point_in_polygon(x, y, polygon_pts: np.ndarray) -> bool:
    contour = polygon_pts.reshape((-1, 1, 2)).astype(np.int32)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def get_zone_label_for_point(x, y):
    for label, poly in zone_polygons.items():
        if point_in_polygon(x, y, poly):
            return label
    return None


def pixel_to_geo(x, y):
    """Transform pixel (x,y) to (lat, lon) using the appropriate ROI."""
    for roi in roi_homographies:
        polygon = roi["pixel_polygon"]
        if cv2.pointPolygonTest(polygon, (x, y), False) >= 0:
            pt = np.array([[x, y, 1.0]], dtype=np.float32).T
            geo = roi["homography"] @ pt
            geo /= geo[2, 0]
            lon, lat = float(geo[0, 0]), float(geo[1, 0])
            return lat, lon
    return None, None


# ==========================
# Device + Models
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

MODELS = {
    "daytime": YOLO(DAY_MODEL_PATH).to(device),
    "nighttime": YOLO(NIGHT_MODEL_PATH).to(device),
}

model, current_model_label = model_selection_criteria(MODELS)
print(f"Initial detection model: {current_model_label}")


# ==========================
# Outputs
# ==========================
cam_num = CAMERA_ID.split("_")[1] if "_" in CAMERA_ID else CAMERA_ID

output_json_path = os.path.join(BASE_DIR, f"vehicle_log_cam_{cam_num}.json")

periodic_log_path = os.path.join(BASE_DIR, f"periodic_counts_cam_{cam_num}.json")
periodic_log_data = []

five_min_log_path = os.path.join(BASE_DIR, f"five_min_counts_cam_{cam_num}.json")
five_min_log_data = []  # optional cumulative

periodic_interval = timedelta(minutes=15)
periodic_interval_start = datetime.now(TZ)
interval_counts = defaultdict(lambda: defaultdict(int))   # {line_name: {class_id: count}}
interval_speeds = defaultdict(lambda: defaultdict(list))  # {line_name: {class_id: [speeds]}}


# ---- Class mapping (YOLO raw -> canonical) ----
CLASS_MAPPING = {
    0: 1,    # person -> pedestrian
    92: 2,   # wheelchair
    1: 3,    # bicycle -> bike
    93: 4,   # e-bike
    91: 5,   # e-scooter
    94: 6,   # skateboard
    95: 7,   # animals
    2: 102,  # car
    7: 103,  # truck
}


def map_class(raw_cls_id: int) -> int:
    return CLASS_MAPPING.get(int(raw_cls_id), int(raw_cls_id))


# ==========================
# Drawing config
# ==========================
roi_color = (50, 50, 50)
line_color = (50, 255, 50)
bbox_color = (50, 50, 255)
trail_color = (255, 255, 255)
center_color = (255, 255, 255)
text_color = (255, 255, 255)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4
thickness = 1


# ==========================
# Initialize VideoCapture + VideoWriter
# ==========================
cap_holder = {"cap": cv2.VideoCapture(input_video_path)}
if not cap_holder["cap"].isOpened():
    raise IOError(f"Cannot open video: {input_video_path}")

fps = int(cap_holder["cap"].get(cv2.CAP_PROP_FPS)) or 30
width = int(cap_holder["cap"].get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
height = int(cap_holder["cap"].get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

# fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# delete_previous_day_clips(BASE_DIR, cam_num)
# output_video_path, current_clip_start = get_output_video_path(BASE_DIR, cam_num, OUTPUT_VIDEO_LENGTH)
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


# ==========================
# Frame Reader Setup
# ==========================
MAX_QUEUE_SIZE = 900
frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
stop_event = threading.Event()


def frame_reader_thread(cap_holder, frame_queue, stop_event):
    """
    Read frames and push them into frame_queue.
    If no frames for > 15 seconds, reinitialize VideoCapture.
    """
    last_good_frame_time = datetime.now(TZ)

    while not stop_event.is_set():
        cap = cap_holder["cap"]
        ret, frame = cap.read()

        if not ret:
            now = datetime.now(TZ)
            offline_secs = (now - last_good_frame_time).total_seconds()
            if offline_secs > 15:
                print("[WARN] No frames for > 15 seconds. Reinitializing VideoCapture...")
                try:
                    cap.release()
                except Exception:
                    pass

                while not stop_event.is_set():
                    new_cap = cv2.VideoCapture(input_video_path)
                    if new_cap.isOpened():
                        cap_holder["cap"] = new_cap
                        print("[INFO] VideoCapture re-opened successfully.")
                        last_good_frame_time = datetime.now(TZ)
                        break
                    print("[WARN] Failed to reopen VideoCapture. Retrying in 5 seconds...")
                    time.sleep(5)
                continue

            time.sleep(0.1)
            continue

        last_good_frame_time = datetime.now(TZ)
        timestamp = last_good_frame_time

        try:
            frame_queue.put_nowait((timestamp, frame))
        except queue.Full:
            pass


reader_thread = threading.Thread(target=frame_reader_thread, args=(cap_holder, frame_queue, stop_event), daemon=True)
reader_thread.start()


# ==========================
# Tracking State
# ==========================
track_history = {}          # track_id -> [(cx, cy)]
geo_history = {}            # track_id -> [(lat, lon, frame)]
vehicle_labels = {}         # track_id -> "Line N"
passed_lines = {}           # track_id -> set(line_name)
line_counters = {line["name"]: 0 for line in counting_lines}
vehicle_pass_log = []
vehicle_speeds = {}
zone_history = {}           # track_id -> list of zones visited

window_size_frames = 15 * 60 * fps
window_start_frame = 0
line_counters_window = {line["name"]: 0 for line in counting_lines}

SPEED_SMOOTHING_WINDOW = 10
speed_history = {}  # track_id -> list of recent speed values

# 5-min window counters
countline_counts = defaultdict(int)   # key: (line_id, direction_label, mapped_cls)
turning_counts = defaultdict(int)     # key: (zone_pair, mapped_cls)
turning_seen = defaultdict(set)       # track_id -> set of (zone_pair, mapped_cls) already counted
current_5min_start = None

countline_ids = defaultdict(set)      # key: (line_id, direction_label, mapped_cls) -> set(track_ids)
turning_ids = defaultdict(set)        # key: (zone_pair, mapped_cls) -> set(track_ids)

last_frame_timestamp = None


# ==========================
# Main Loop
# ==========================
mem_info = process.memory_info()
print(f"[INIT] Memory Usage (RSS): {mem_info.rss / (1024 ** 2):.2f} MB")

frame_idx = 0

try:
    while True:
        try:
            frame_timestamp, frame = frame_queue.get(timeout=1)
            last_frame_timestamp = frame_timestamp
        except queue.Empty:
            print("Frame queue empty — waiting for frames...")
            continue

        frame_idx += 1

        # --- 5-min window management (aligned to wall-clock) ---
        window_start = floor_to_5min(frame_timestamp)
        if current_5min_start is None:
            current_5min_start = window_start
        elif window_start != current_5min_start:
            window_end = window_start

            window_entries = []
            flush_five_min_window(
                start_dt=current_5min_start,
                end_dt=window_end,
                countline_ids=countline_ids,
                turning_ids=turning_ids,
                out_entries=window_entries,
            )

            five_min_log_data.extend(window_entries)

            with open(five_min_log_path, "w") as f:
                json.dump(convert_numpy_types(window_entries), f, indent=4)

            send_five_min_log(five_min_log_path)

            countline_counts.clear()
            turning_counts.clear()
            turning_seen.clear()
            countline_ids.clear()
            turning_ids.clear()
            current_5min_start = window_start

        # Print buffer size every 100 frames
        if frame_idx % 100 == 0:
            print(f"[INFO] Frame queue size: {frame_queue.qsize()}")
            mem_info = process.memory_info()
            print(f"[INFO] Memory Usage (RSS): {mem_info.rss / (1024 ** 2):.2f} MB")
            if not frame_queue.empty():
                frame_size = frame_queue.queue[0][1].nbytes
                total_buffer_bytes = frame_queue.qsize() * frame_size
                print(f"[INFO] Frame buffer size: {total_buffer_bytes / (1024 ** 2):.2f} MB")

        # Switch day/night model approx hourly
        if frame_idx % max(int(fps * 3600), 1) == 0:
            new_model, new_label = model_selection_criteria(MODELS)
            if new_model is not model:
                model = new_model
                current_model_label = new_label
                print(f"[{datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}] Switched detection model to: {current_model_label}")

        # Inference/tracking
        results = model.track(frame, device=device, persist=True, verbose=False)

        # Draw ROIs (if you want visible boundaries)
        for poly in roi_polygons_for_draw:
            pts = poly.reshape((-1, 1, 2)).astype(int)
            # cv2.polylines(frame, [pts], isClosed=True, color=roi_color, thickness=1)

        # Draw zone polygons
        for z_label, poly in zone_polygons.items():
            pts = poly.reshape((-1, 1, 2)).astype(int)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            x_txt, y_txt = int(poly[0, 0]), int(poly[0, 1])
            cv2.putText(frame, f"Zone {z_label}", (x_txt, y_txt - 5), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw counting lines
        for line in counting_lines:
            (x1, y1), (x2, y2) = line["points"]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), line_color, 1)
            cv2.putText(frame, f"{line['name']}", (int(x1) + 5, int(y1) - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

        # Handle detections
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            ids = results[0].boxes.id
            ids = ids.cpu().numpy().astype(int) if ids is not None else [None] * len(boxes)

            for box, cls_id, track_id in zip(boxes, classes, ids):
                if track_id is None:
                    continue

                raw_cls_id = int(cls_id)
                mapped_cls = map_class(raw_cls_id)

                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int(max(y1, y2))

                lat, lon = pixel_to_geo(cx, cy)
                if lat is None or lon is None:
                    continue

                label = model.names[raw_cls_id] if raw_cls_id in model.names else str(raw_cls_id)

                # Zone tagging (skip ped/wheelchair/animals)
                if mapped_cls not in (1, 2, 7):
                    current_zone = get_zone_label_for_point(cx, cy)
                else:
                    current_zone = None

                if current_zone is not None:
                    hist = zone_history.setdefault(track_id, [])
                    if not hist or hist[-1] != current_zone:
                        hist.append(current_zone)

                zone_tag = ""
                hist = zone_history.get(track_id, [])
                if len(hist) >= 1:
                    first_zone = hist[0]
                    last_zone = hist[-1]
                    zone_tag = first_zone if first_zone == last_zone else (first_zone + last_zone)

                # Turning movement counting (zone_pair, mapped_cls)
                if len(zone_tag) == 2 and mapped_cls not in (1, 2, 7):
                    tm_key = (zone_tag, mapped_cls)
                    seen_set = turning_seen.setdefault(track_id, set())
                    if tm_key not in seen_set:
                        turning_counts[tm_key] += 1
                        turning_ids[tm_key].add(int(track_id))
                        seen_set.add(tm_key)

                # Draw detection visuals
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 1)
                cv2.circle(frame, (cx, cy), 2, center_color, -1)

                # Update history
                track_history.setdefault(track_id, []).append((cx, cy))
                geo_history.setdefault(track_id, []).append((lat, lon, frame_idx))

                if len(track_history[track_id]) > 20:
                    track_history[track_id].pop(0)
                if len(geo_history[track_id]) > 20:
                    geo_history[track_id].pop(0)

                # Draw short motion trail
                if len(track_history[track_id]) >= 5:
                    prev_point = track_history[track_id][-5]
                    curr_point = (cx, cy)
                    cv2.line(frame, prev_point, curr_point, trail_color, 1)

                    # Check line intersection
                    for line in counting_lines:
                        p1, p2 = line["points"]
                        A, B = (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))
                        line_id = str(line["name"])

                        # Estimate speed
                        smoothed_speed = None
                        if len(geo_history[track_id]) >= 2:
                            lat1, lon1, f1 = geo_history[track_id][-2]
                            lat2, lon2, f2 = geo_history[track_id][-1]
                            dist_km = haversine(lat1, lon1, lat2, lon2)
                            delta_t = (f2 - f1) / fps
                            if delta_t > 0:
                                speed_kmh = (dist_km / delta_t) * 3600
                                speed_mph = speed_kmh * 0.621371
                                speed_history.setdefault(track_id, []).append(speed_mph)
                                if len(speed_history[track_id]) > SPEED_SMOOTHING_WINDOW:
                                    speed_history[track_id].pop(0)
                                smoothed_speed = sum(speed_history[track_id]) / len(speed_history[track_id])
                                vehicle_speeds[track_id] = smoothed_speed

                        # Exclude cars/trucks from line counting
                        if lines_intersect(prev_point, curr_point, A, B) and mapped_cls not in (102, 103):
                            # Direction via cross product
                            lx = B[0] - A[0]
                            ly = B[1] - A[1]
                            mx = curr_point[0] - prev_point[0]
                            my = curr_point[1] - prev_point[1]
                            z = lx * my - ly * mx
                            direction_label = "C" if z > 0 else "CC"

                            cv2.circle(frame, curr_point, radius=5, color=(0, 0, 255), thickness=-1)

                            passed_lines.setdefault(track_id, set())
                            if line["name"] not in passed_lines[track_id]:
                                passed_lines[track_id].add(line["name"])

                                vehicle_labels[track_id] = f"Line {line_id} {direction_label}"
                                line_counters[line["name"]] += 1
                                line_counters_window[line["name"]] += 1

                                vehicle_pass_log.append({
                                    "Timestamp": frame_timestamp.isoformat(),
                                    "FrameID": frame_idx,
                                    "ClassID": int(cls_id),
                                    "TrackID": int(track_id),
                                    "Latitude": lat,
                                    "Longitude": lon,
                                    "Speed_mph": smoothed_speed,
                                    "Direction": direction_label,
                                    "LaneNumber": line_id
                                })

                                interval_counts[line["name"]][cls_id] += 1
                                if smoothed_speed is not None:
                                    interval_speeds[line["name"]][cls_id].append(smoothed_speed)

                                key = (line_id, direction_label, mapped_cls)
                                countline_counts[key] += 1
                                countline_ids[key].add(int(track_id))
                                
                main_label = f"{label} ID:{track_id}"
                if zone_tag:
                    main_label += f" [{zone_tag}]"
                if track_id in vehicle_labels:
                    main_label += f" ({vehicle_labels[track_id]})"
                    
                cv2.putText(frame, main_label, (x1, max(y1 - 10, 20)), font, font_scale, text_color, thickness, cv2.LINE_AA)
                
                if track_id in vehicle_speeds:
                    speed_label = f"{vehicle_speeds[track_id]:.1f} mph"
                    cv2.putText(frame, speed_label, (x1, y2 + 15), font, 0.4, (50, 255, 50), 1, cv2.LINE_AA)
        	

        # Periodic logging (every 15 minutes)
        now = datetime.now(TZ)
        if now >= periodic_interval_start + periodic_interval:
            interval_end = now
            for line_name in list(interval_counts.keys()):
                for cls_id, count in interval_counts[line_name].items():
                    speeds = interval_speeds[line_name][cls_id]
                    avg_speed = (sum(speeds) / len(speeds)) if speeds else 0.0
                    direction = next((ln.get("direction", "N/A") for ln in counting_lines if ln["name"] == line_name), "N/A")
                    lane_number = next((i + 1 for i, ln in enumerate(counting_lines) if ln["name"] == line_name), -1)

                    periodic_log_data.append({
                        "Start_date_time": periodic_interval_start.strftime("%Y-%m-%d %H:%M:%S"),
                        "End_date_time": interval_end.strftime("%Y-%m-%d %H:%M:%S"),
                        "LaneNumber": lane_number,
                        "Count": int(count),
                        "ClassID": int(cls_id),
                        "avg_speed_mph": round(float(avg_speed), 2),
                        "Direction": direction,
                    })

            with open(periodic_log_path, "w") as f:
                json.dump(convert_numpy_types(periodic_log_data), f, indent=4)

            periodic_interval_start = interval_end
            interval_counts.clear()
            interval_speeds.clear()

        # Reset the display window every 15 minutes (visual counters)
        if frame_idx - window_start_frame >= window_size_frames:
            window_start_frame = frame_idx
            line_counters_window = {line["name"]: 0 for line in counting_lines}

        # ---------- Draw 5-min per-(countline, zone, class) table (top-right) ----------
        table_x = width - 360
        table_y = 30

        rows = []
        for (line_id, dir_label, mode_id), cnt in sorted(countline_counts.items()):
            rows.append(f"Count Line: {line_id}({dir_label}) | mode{mode_id} | {cnt}")

        for (zone_pair, mode_id), cnt in sorted(turning_counts.items()):
            rows.append(f"Turning Movement: {zone_pair} | mode{mode_id} | {cnt}")

        if current_5min_start is not None:
            window_end = current_5min_start + timedelta(minutes=5)
            header1 = f"5-min: {current_5min_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')}"
        else:
            header1 = "5-min: N/A"
        header2 = f"Now: {frame_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

        line_height = 22
        num_lines = 2 + 1 + len(rows)
        box_top = table_y - 25
        box_bottom = table_y + num_lines * line_height

        cv2.rectangle(frame, (table_x - 10, box_top), (width - 10, box_bottom), (40, 40, 40), -1)

        cv2.putText(frame, "5-min Counts (line / zone / class)", (table_x, table_y),
                    font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        y_cursor = table_y + line_height
        cv2.putText(frame, header1, (table_x, y_cursor), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        y_cursor += line_height
        cv2.putText(frame, header2, (table_x, y_cursor), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        for row in rows:
            y_cursor += line_height
            cv2.putText(frame, row, (table_x, y_cursor), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Write + display
        # out.write(frame)
        # cv2.imshow("Tracking", frame)

        # Rotate clip every OUTPUT_VIDEO_LENGTH minutes
        # if now >= current_clip_start + timedelta(minutes=OUTPUT_VIDEO_LENGTH):
            # out.release()
            # output_video_path, current_clip_start = get_output_video_path(BASE_DIR, cam_num, OUTPUT_VIDEO_LENGTH)
            # out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Exit on 'q'
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

except KeyboardInterrupt:
    print("Interrupted by user (Ctrl+C) — exiting...")

finally:
    # Stop reader thread
    stop_event.set()
    try:
        reader_thread.join(timeout=5)
    except Exception:
        pass

    # Release capture + writer
    try:
        if cap_holder.get("cap") is not None:
            cap_holder["cap"].release()
    except Exception:
        pass

    # try:
        # out.release()
    except Exception:
        pass

    # try:
    #     cv2.destroyAllWindows()
    except Exception:
        pass

    # Flush last (possibly partial) 5-min window
    if current_5min_start is not None and last_frame_timestamp is not None:
        last_window_start = floor_to_5min(last_frame_timestamp)
        last_window_end = last_window_start + timedelta(minutes=5)

        last_window_entries = []
        flush_five_min_window(
            start_dt=current_5min_start,
            end_dt=last_window_end,
            countline_ids=countline_ids,
            turning_ids=turning_ids,
            out_entries=last_window_entries,
        )

        five_min_log_data.extend(last_window_entries)

        with open(five_min_log_path, "w") as f:
            json.dump(convert_numpy_types(last_window_entries), f, indent=4)
        send_five_min_log(five_min_log_path)

    with open(output_json_path, "w") as f:
        json.dump(convert_numpy_types(vehicle_pass_log), f, indent=4)

    with open(periodic_log_path, "w") as f:
        json.dump(convert_numpy_types(periodic_log_data), f, indent=4)

    with open(five_min_log_path.replace(".json", "_all.json"), "w") as f:
        json.dump(convert_numpy_types(five_min_log_data), f, indent=4)

    mem_info = process.memory_info()
    print(f"[DONE] Memory Usage (RSS): {mem_info.rss / (1024 ** 2):.2f} MB")
    print(f"✅ Processed {frame_idx} frames.")
    print(f"🧾 JSON log with geospatial speeds: {output_json_path}")
    # print(f"✅ Finished processing. Last clip: {output_video_path}")
    print(f"🧾 15-min interval log saved to: {periodic_log_path}")
    print(f"🧾 5-min log saved to: {five_min_log_path}")
