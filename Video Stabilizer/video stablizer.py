import cv2
import numpy as np
import os
from tqdm import tqdm

raw_video_dir = "raw_video"
output_dir = "stabilized_clips"

os.makedirs(output_dir, exist_ok=True)

def moving_average_curve(curve, radius):
    window_size = 2 * radius + 1
    filter_kernel = np.ones(window_size) / window_size
    curve_pad = np.pad(curve, (radius, radius), "edge")
    smoothed = np.convolve(curve_pad, filter_kernel, mode="same")
    return smoothed[radius: -radius]

def smooth_trajectory(trajectory, smoothing_radius=30):
    smoothed = np.copy(trajectory)
    for i in range(3):
        smoothed[:, i] = moving_average_curve(trajectory[:, i], radius=smoothing_radius)
    return smoothed

def stabilize_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    prev_gray = None
    transforms = []

    for _ in tqdm(range(n_frames - 1), desc=f"Analyzing {os.path.basename(input_path)}"):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
            continue
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
        valid_prev = prev_pts[status == 1]
        valid_curr = curr_pts[status == 1]
        m, _ = cv2.estimateAffinePartial2D(valid_prev, valid_curr)
        if m is None:
            transforms.append([0, 0, 0])
            continue
        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
        transforms.append([dx, dy, da])
        prev_gray = gray
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)

    transforms = np.array(transforms)
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth_trajectory(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in tqdm(range(len(transforms_smooth)), desc=f"Stabilizing {os.path.basename(input_path)}"):
        ret, frame = cap.read()
        if not ret:
            break
        dx, dy, da = transforms_smooth[i]
        m = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da),  np.cos(da), dy]
        ])
        stabilized = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out.write(stabilized)

    cap.release()
    out.release()

for filename in os.listdir(raw_video_dir):
    if filename.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
        input_path = os.path.join(raw_video_dir, filename)
        output_path = os.path.join(output_dir, f"stabilized_{filename}")
        stabilize_video(input_path, output_path)

print("✅ Stabilization completed.")
