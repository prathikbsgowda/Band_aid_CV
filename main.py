#!/usr/bin/env python3
"""
Band-aid overlay tool for a healthcare assignment.

Usage example:
python main.py --input images/sample_arm.jpg --bandaid images/bandaid.png --output results/result.jpg --method mediapipe
"""
import argparse
import os
import cv2
import numpy as np

# Try to import mediapipe (optional but recommended)
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

def detect_arm_mediapipe(img):
    """Return (shoulder_px, elbow_px, wrist_px) in pixel coordinates if found, else None."""
    if not HAS_MEDIAPIPE:
        return None
    h, w = img.shape[:2]
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark
        # choose arm (left or right) by combined visibility of elbow & wrist
        left_vis = (lm[mp_pose.PoseLandmark.LEFT_ELBOW].visibility +
                    lm[mp_pose.PoseLandmark.LEFT_WRIST].visibility)
        right_vis = (lm[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility +
                     lm[mp_pose.PoseLandmark.RIGHT_WRIST].visibility)
        if right_vis >= left_vis:
            shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
        else:
            shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
            wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
        def to_px(lm_obj):
            return int(lm_obj.x * w), int(lm_obj.y * h)
        return to_px(shoulder), to_px(elbow), to_px(wrist)

def detect_arm_skin_contour(img):
    """
    Fallback simple skin-color segmentation.
    Returns (pt1, center, pt2) where pt1/pt2 are points along the long axis.
    """
    h, w = img.shape[:2]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Broad skin range (may need tuning for diverse skin tones)
    lower = np.array([0, 20, 70])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(img_hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    center = (int(rect[0][0]), int(rect[0][1]))
    width, height = int(rect[1][0]), int(rect[1][1])
    # Return two endpoints along the major axis
    angle = rect[2]
    if width >= height:
        half = int(width/2)
        dx = int(np.cos(np.deg2rad(angle)) * half)
        dy = int(np.sin(np.deg2rad(angle)) * half)
    else:
        half = int(height/2)
        dx = int(np.cos(np.deg2rad(angle+90)) * half)
        dy = int(np.sin(np.deg2rad(angle+90)) * half)
    pt1 = (center[0]-dx, center[1]-dy)
    pt2 = (center[0]+dx, center[1]+dy)
    return pt1, center, pt2

def rotate_and_scale_overlay(overlay, angle_deg, scale):
    """
    Rotate and scale an RGBA overlay and return the rotated image (still RGBA).
    overlay: numpy array with 4 channels (BGRA or RGBA depending on cv2 read mode).
    """
    h, w = overlay.shape[:2]
    new_w = max(1, int(w*scale))
    new_h = max(1, int(h*scale))
    overlay_resized = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
    (cX, cY) = (new_w//2, new_h//2)
    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    nW = int((new_h * sin) + (new_w * cos))
    nH = int((new_h * cos) + (new_w * sin))
    M[0,2] += (nW/2) - cX
    M[1,2] += (nH/2) - cY
    rotated = cv2.warpAffine(overlay_resized, M, (nW, nH), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

def overlay_image_alpha(bg, overlay, center):
    """
    Composite RGBA overlay onto BGR bg at center (x,y).
    bg: BGR uint8 image
    overlay: BGRA/rgba (4 channel) uint8 image
    center: (x,y) center location in bg where overlay will be placed
    """
    x = int(center[0] - overlay.shape[1] / 2)
    y = int(center[1] - overlay.shape[0] / 2)
    h, w = bg.shape[:2]
    ow = overlay.shape[1]
    oh = overlay.shape[0]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x+ow), min(h, y+oh)
    if x1 >= x2 or y1 >= y2:
        return bg
    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)
    roi = bg[y1:y2, x1:x2].astype(float)
    arr_overlay = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2].astype(float)
    # Ensure overlay has alpha channel
    if arr_overlay.shape[2] < 4:
        alpha = np.ones((arr_overlay.shape[0], arr_overlay.shape[1],1))*255.0
        arr_overlay = np.concatenate([arr_overlay,alpha],axis=2)
    alpha = arr_overlay[...,3:]/255.0
    foreground = arr_overlay[...,:3]
    blended = alpha*foreground + (1-alpha)*roi
    bg[y1:y2, x1:x2] = blended.astype(np.uint8)
    return bg

def place_band_aid_on_arm(img_bgr, bandaid_path, arm_points):
    """
    Place band-aid centered on the midpoint between two chosen arm points.
    arm_points: if from mediapipe: (shoulder, elbow, wrist) else (pt1, center, pt2)
    """
    if len(arm_points) == 3:
        p1, p2 = arm_points[1], arm_points[2]  # elbow -> wrist (so band-aid sits on forearm)
    else:
        p1, p2 = arm_points[0], arm_points[2]
    p1 = np.array(p1); p2 = np.array(p2)
    mid = ((p1 + p2)/2).astype(int)
    dx, dy = (p2 - p1)[0], (p2 - p1)[1]
    angle = np.degrees(np.arctan2(dy, dx))
    distance = np.linalg.norm(p2 - p1)
    overlay = cv2.imread(bandaid_path, cv2.IMREAD_UNCHANGED)  # BGRA expected
    if overlay is None:
        raise FileNotFoundError(f"Could not read bandaid image: {bandaid_path}")
    bw = overlay.shape[1]
    # scale so bandaid roughly covers a proportion of detected arm width
    scale = max(0.15, (distance * 0.9) / max(1, bw))
    rotated = rotate_and_scale_overlay(overlay, angle, scale)
    out = overlay_image_alpha(img_bgr, rotated, (int(mid[0]), int(mid[1])))
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--bandaid", required=True, help="Path to bandaid PNG (with alpha)")
    parser.add_argument("--output", default="results/result.jpg", help="Output path")
    parser.add_argument("--method", choices=["mediapipe","skin"], default="mediapipe")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    img = cv2.imread(args.input)
    if img is None:
        print("Could not read input image:", args.input)
        return
    arm_points = None
    if args.method=="mediapipe":
        arm_points = detect_arm_mediapipe(img)
        if arm_points is None:
            print("Mediapipe failed to find arm; falling back to skin-based detection")
            arm_points = detect_arm_skin_contour(img)
    else:
        arm_points = detect_arm_skin_contour(img)
    if arm_points is None:
        print("Could not detect arm in the image. Try a clearer image or use different method.")
        return
    out = place_band_aid_on_arm(img.copy(), args.bandaid, arm_points)
    # side-by-side for easier visualization
    try:
        side = np.hstack([img, out])
    except Exception:
        out_resized = cv2.resize(out, (img.shape[1], img.shape[0]))
        side = np.hstack([img, out_resized])
    cv2.imwrite(args.output, side)
    print(f"Saved result to {args.output}")

if __name__=="__main__":
    main()
