# Advanced Computer Graphics & Digital Image Processing Suite

An interactive desktop application built using Python, Tkinter, PyOpenGL, and OpenCV. This project integrates fundamental 2D and 3D graphics rendering with spatial and morphological digital image processing methods and an automated medical image ROI analysis pipeline.

## Project Overview

Course:  Graphics &  Image Processing Lab
Submitted By: Nurer Nahar (ID: 20235203011)
Department: Department of Computer Science & Engineering
Institution: Bangladesh University of Business and Technology (BUBT)

## Key Modules & Technical Features

1. Computer Graphics
- 2D Transformations: Translation, Rotation, Scaling, Shearing, and Reflection using custom matrix calculations.
- 3D Projection: Interactive 3D object rendering with view controls using PyOpenGL.

2. Digital Image Processing (DIP)
- Spatial Domain Filtering: Smoothing, Sharpening, Edge Detection (Sobel/Laplacian), and Custom Convolutions.
- Morphological Operations: Erosion, Dilation, Opening, and Closing for structural image analysis.
- Color Conversions: Real-time color space transformations between RGB, Grayscale, and HSV.

3. Automated Medical ROI Pipeline
- Automated segmentation and Region of Interest (ROI) extraction on brain MRI scans to highlight structural anomalies using adaptive thresholding and contour processing.

## Project Structure

- main.py (Primary execution entry point)
- modules/
  - basic_filters.py
  - spatial_filters.py
  - morphology.py
  - transformations_2d.py
  - graphics_2d.py
  - medical_pipeline.py
  - app_gui.py
- final output ss.png.png
- README.md

## Environment Setup & Installation

Ensure you have Python 3.8+ installed on your system.

1. Clone the Repository:
`git clone https://github.com/NurNaharjhumur/Computer-Graphics-DIP-Toolkit.git`
`cd Computer-Graphics-DIP-Toolkit`

2. Install Required Dependencies:
`pip install opencv-python numpy matplotlib PyOpenGL PyOpenGL_accelerate`

3. Run the Application:
`python main.py`
