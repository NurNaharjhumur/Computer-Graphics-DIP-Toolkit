import cv2
import numpy as np

INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

def draw_clipping_window(img):
    if img is not None:
        h, w = img.shape[:2]
        xmin, ymin, xmax, ymax = int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
    return img

def execute_cohen_sutherland(base_img, user_lines):
    if base_img is None or not user_lines:
        return base_img, 0

    processed_img = base_img.copy()
    h, w = processed_img.shape[:2]
    xmin, ymin, xmax, ymax = int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)

    def compute_code(x, y):
        code = INSIDE
        if x < xmin: code |= LEFT
        elif x > xmax: code |= RIGHT
        if y < ymin: code |= BOTTOM
        elif y > ymax: code |= TOP
        return code

    cv2.rectangle(processed_img, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
    clipped_count = 0

    for line in user_lines:
        x1, y1, x2, y2 = line
        code1 = compute_code(x1, y1)
        code2 = compute_code(x2, y2)
        accept = False

        while True:
            if code1 == 0 and code2 == 0:
                accept = True
                break
            elif (code1 & code2) != 0:
                break
            else:
                x, y = 0.0, 0.0
                outcode = code1 if code1 != 0 else code2

                if outcode & TOP:
                    x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
                    y = ymax
                elif outcode & BOTTOM:
                    x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
                    y = ymin
                elif outcode & RIGHT:
                    y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
                    x = xmax
                elif outcode & LEFT:
                    y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
                    x = xmin

                if outcode == code1:
                    x1, y1 = int(x), int(y)
                    code1 = compute_code(x1, y1)
                else:
                    x2, y2 = int(x), int(y)
                    code2 = compute_code(x2, y2)

        if accept:
            cv2.line(processed_img, (x1, y1), (x2, y2), (0, 255, 0), 4)
            clipped_count += 1

    return processed_img, clipped_count
