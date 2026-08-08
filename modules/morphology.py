import cv2
import numpy as np

def apply_erosion(img):
    if img is not None:
        kernel = np.ones((3,3), np.uint8)
        return cv2.erode(img, kernel, iterations=1)
    return img

def apply_dilation(img):
    if img is not None:
        kernel = np.ones((3,3), np.uint8)
        return cv2.dilate(img, kernel, iterations=1)
    return img

def apply_opening(img):
    if img is not None:
        kernel = np.ones((3,3), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img

def apply_closing(img):
    if img is not None:
        kernel = np.ones((3,3), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img
