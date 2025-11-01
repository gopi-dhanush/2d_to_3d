
import cv2
import numpy as np
from main_of_3d.converter_3d_2 import estimate_depth
def generate_stereo(frame, depth_norm, shift=10):
    h, w = frame.shape[:2]
    left = np.zeros_like(frame)
    right = np.zeros_like(frame)

    for y in range(h):
        for x in range(w):
            dx = int(shift * depth_norm[y, x])  # bigger shift for closer pixels
            if x - dx >= 0:
                left[y, x - dx] = frame[y, x]
            if x + dx < w:
                right[y, x + dx] = frame[y, x]

    return np.hstack((left, right))  # side-by-side stereo
def generate_anaglyph(left, right):
    anaglyph = np.zeros_like(left)
    anaglyph[..., 0] = left[..., 0]   # Red from left
    anaglyph[..., 1] = right[..., 1]  # Green from right
    anaglyph[..., 2] = right[..., 2]  # Blue from right
    return anaglyph
if depth_map is None:
    print("error at depth estimation")
else:
    # Normalize for visualization
    depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
out = cv2.VideoWriter("output/3d_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 60, (2*w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    depth = estimate_depth(frame)
    depth_norm = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
    stereo = generate_stereo(frame, depth_norm)
    out.write(stereo)
