import cv2
import numpy as np

def run_xray_fracture_pipeline(img):
    if img is None:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    blurred = cv2.medianBlur(gray, 5)

    _, thresh = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        img_area = img.shape[0] * img.shape[1]

        if (img_area * 0.005) < area < (img_area * 0.15) and circularity > 0.3:
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            result_bgr[mask > 0] = [0, 0, 255]
            
            cv2.drawContours(result_bgr, [cnt], -1, (0, 255, 0), 2)

    return result_bgr
