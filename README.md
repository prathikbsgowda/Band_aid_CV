# 🩹 Band-Aid Overlay on Arm (Healthcare Assignment)

This project demonstrates a simple **computer vision healthcare application**:  
placing a **digital band-aid on a person’s arm** from an input image.

The program supports **two arm detection methods**:
1. **Mediapipe Pose Landmarks** – AI-based body landmark detection.
2. **Skin-Color Segmentation** – Basic HSV color-based contour detection.

---

## 📂 Project Structure
```
bandage_assignment/
│── images/          # Input images (arms, band-aid PNG)
│── results/         # Output images
│── main.py          # Main program
│── requirements.txt # Dependencies
│── README.md        # Project instructions
```

---

## ⚙️ Requirements

Install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Dependencies:**
- opencv-python
- numpy
- mediapipe
- Pillow

---

## ▶️ How to Run

### Using Mediapipe (preferred)
```bash
python main.py --input images/sample_arm.jpg --bandaid images/bandaid.png --output results/out_mediapipe.jpg --method mediapipe
```

### Using Skin-Color Segmentation (fallback)
```bash
python main.py --input images/sample_arm.jpg --bandaid images/bandaid.png --output results/out_skin.jpg --method skin
```

---

## 📊 Output

The program outputs a side-by-side comparison:
- Left → Original input image
- Right → Band-aid digitally applied on the arm  

Example:
```
results/
│── out_mediapipe.jpg   # Result using Mediapipe
│── out_skin.jpg        # Result using Skin detection
```

---

## 🧠 Methods Explained

### 1. Mediapipe (AI Landmark Detection)
- Detects shoulder, elbow, wrist using Mediapipe Pose.
- Band-aid is scaled and rotated to match the forearm angle.
- ✅ Works across different skin tones, backgrounds, and lighting.
- 🔻 Requires Mediapipe installed.

### 2. Skin-Color Segmentation
- Converts image to HSV color space.
- Detects skin regions with a fixed color range.
- Band-aid is placed at the arm’s contour center.
- ✅ Lightweight, no heavy model.
- 🔻 May fail for darker/lighter tones or poor lighting.

---

## 🩺 Healthcare Context

This assignment simulates **medical image augmentation**.  
Applications include:
- Training healthcare AI systems on wound care or bandage placement.
- Augmented reality in remote consultations.
- Educational tools for nursing & first-aid training.

⚠️ **Disclaimer:** This project is for educational/demo purposes only and not for real clinical use.

---

## 🙌 Credits
- OpenCV (image processing)  
- Mediapipe (pose estimation)  
- NumPy (math operations)
