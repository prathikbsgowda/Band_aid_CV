# Approach Note

The goal of this project was to digitally place a band-aid on a person’s arm in an input image, making it appear natural and healthcare-specific. The solution was implemented in Python using computer vision libraries. Two different approaches were considered for detecting the arm:

1. **Skin-color based segmentation** using OpenCV. By converting the image to HSV color space, a skin-tone range was defined, and contours were extracted to isolate the arm region. This method works well in controlled lighting but can be sensitive to skin tone variations and shadows.

2. **Mediapipe Pose estimation** for robust detection. Mediapipe provides pre-trained models that detect human body landmarks. Using arm landmarks, the approximate position and orientation of the arm were located. This allowed dynamic placement of the band-aid at the correct angle and scale.

Once the arm was detected, a transparent PNG of a band-aid was overlaid onto the image. OpenCV functions (`cv2.warpAffine` and `cv2.addWeighted`) were used to resize, rotate, and blend the band-aid so that it aligned naturally with the arm.

**Assumptions** include:  
- Input images are reasonably well-lit  
- The arm is visible and unobstructed  
- The band-aid should be applied in a fixed region of the upper arm for demonstration  

This project demonstrates how computer vision techniques can be applied in healthcare-related image augmentation tasks, which can further be extended for virtual try-on applications or patient education tools.
