import cv2
import numpy as np
import matplotlib.pyplot as plt

def to_grayscale(img):
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def to_binary(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary
    return img

def apply_histogram_equalization(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return cv2.equalizeHist(gray)
    return img

def plot_histogram(orig_img, proc_img):
    if orig_img is None or proc_img is None:
        return False

    orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY) if len(orig_img.shape) == 3 else orig_img
    proc_gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY) if len(proc_img.shape) == 3 else proc_img

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(orig_gray.ravel(), 256, [0, 256], color='blue')
    plt.title('Original Image Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(proc_gray.ravel(), 256, [0, 256], color='green')
    plt.title('Processed Image Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    return True

def split_rgb_channels(img):
    if img is not None:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
        b, g, r = cv2.split(img_bgr)
        zeros = np.zeros_like(b)
        red_ch = cv2.merge([zeros, zeros, r])
        green_ch = cv2.merge([zeros, g, zeros])
        blue_ch = cv2.merge([b, zeros, zeros])
        return np.hstack((blue_ch, green_ch, red_ch))
    return img

def to_depth_map(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return img

def to_negative(img):
    if img is not None:
        return cv2.bitwise_not(img)
    return img
