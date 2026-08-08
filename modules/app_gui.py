import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# Import logic modules from current package
from modules import basic_filters as bf
from modules import spatial_filters as sf
from modules import morphology as morph
from modules import graphics_2d as g2d
from modules import transformations_2d as t2d
from modules import medical_pipeline as med

class CompleteGraphicsDIPSuite:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Computer Graphics & DIP Complete Suite")
        self.root.geometry("1520x920")
        self.root.configure(bg="#1e1e24")

        self.original_img = None
        self.processed_img = None

        self.draw_mode = None  
        self.start_x = None
        self.start_y = None
        self.user_lines = []

        self.disp_w = 640
        self.disp_h = 480

        self._setup_ui()

    def _setup_ui(self):
        # --- Toolbar Row 1 ---
        self.main_container = tk.Frame(self.root, bg="#1e1e24", padx=10, pady=5)
        self.main_container.pack(side=tk.TOP, fill=tk.X)

        grp_file = tk.LabelFrame(self.main_container, text=" File Operations ", bg="#2b2d42", fg="#4cc9f0", font=('Helvetica', 8, 'bold'), padx=5, pady=2)
        grp_file.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        tk.Button(grp_file, text="Load Image", command=self.load_image, bg="#3a0ca3", fg="white").pack(side=tk.LEFT, padx=1)
        tk.Button(grp_file, text="Save Image", command=self.save_image, bg="#7209b7", fg="white").pack(side=tk.LEFT, padx=1)
        tk.Button(grp_file, text="Reset Canvas", command=self.reset_image, bg="#f72585", fg="white").pack(side=tk.LEFT, padx=1)

        grp_color = tk.LabelFrame(self.main_container, text=" Color, Histogram & Depth ", bg="#2b2d42", fg="#4cc9f0", font=('Helvetica', 8, 'bold'), padx=5, pady=2)
        grp_color.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        tk.Button(grp_color, text="Gray", command=self.cmd_grayscale).pack(side=tk.LEFT, padx=1)
        tk.Button(grp_color, text="Threshold", command=self.cmd_binary).pack(side=tk.LEFT, padx=1)
        tk.Button(grp_color, text="Hist Equalize", command=self.cmd_hist_equalize, bg="#e76f51", fg="white").pack(side=tk.LEFT, padx=1)
        tk.Button(grp_color, text="Plot Histogram", command=self.cmd_plot_hist, bg="#2a9d8f", fg="white").pack(side=tk.LEFT, padx=1)
        tk.Button(grp_color, text="RGB Split", command=self.cmd_rgb_split, bg="#4361ee", fg="white").pack(side=tk.LEFT, padx=1)
        tk.Button(grp_color, text="2D->3D Depth", command=self.cmd_depth_map, bg="#7209b7", fg="white").pack(side=tk.LEFT, padx=1)
        tk.Button(grp_color, text="Negative", command=self.cmd_negative).pack(side=tk.LEFT, padx=1)

        grp_filter = tk.LabelFrame(self.main_container, text=" Filters & Frequency Domain ", bg="#2b2d42", fg="#ffb703", font=('Helvetica', 8, 'bold'), padx=5, pady=2)
        grp_filter.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        tk.Button(grp_filter, text="Gaussian Blur", command=self.cmd_gaussian).pack(side=tk.LEFT, padx=1)
        tk.Button(grp_filter, text="Median Filter", command=self.cmd_median).pack(side=tk.LEFT, padx=1)
        tk.Button(grp_filter, text="Sharpen", command=self.cmd_sharpen).pack(side=tk.LEFT, padx=1)
        tk.Button(grp_filter, text="Sobel (1st)", command=self.cmd_sobel).pack(side=tk.LEFT, padx=1)
        tk.Button(grp_filter, text="Laplacian (2nd)", command=self.cmd_laplacian).pack(side=tk.LEFT, padx=1)
        tk.Button(grp_filter, text="Canny Edge", command=self.cmd_canny, bg="#2a9d8f", fg="white").pack(side=tk.LEFT, padx=1)
        tk.Button(grp_filter, text="FFT Low-Pass", command=lambda: self.cmd_fft('low'), bg="#3a0ca3", fg="white").pack(side=tk.LEFT, padx=1)
        tk.Button(grp_filter, text="FFT High-Pass", command=lambda: self.cmd_fft('high'), bg="#7209b7", fg="white").pack(side=tk.LEFT, padx=1)

        # --- Toolbar Row 2 ---
        self.sub_container = tk.Frame(self.root, bg="#1e1e24", padx=10, pady=2)
        self.sub_container.pack(side=tk.TOP, fill=tk.X)

        grp_morph = tk.LabelFrame(self.sub_container, text=" Morphology ", bg="#2b2d42", fg="#80ed99", font=('Helvetica', 8, 'bold'), padx=5, pady=2)
        grp_morph.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        tk.Button(grp_morph, text="Erosion", command=self.cmd_erosion).pack(side=tk.LEFT, padx=1)
        tk.Button(grp_morph, text="Dilation", command=self.cmd_dilation).pack(side=tk.LEFT, padx=1)
        tk.Button(grp_morph, text="Opening", command=self.cmd_opening).pack(side=tk.LEFT, padx=1)
        tk.Button(grp_morph, text="Closing", command=self.cmd_closing).pack(side=tk.LEFT, padx=1)

        grp_cg = tk.LabelFrame(self.sub_container, text=" Interactive CG & Clipping ", bg="#2b2d42", fg="#f72585", font=('Helvetica', 8, 'bold'), padx=5, pady=2)
        grp_cg.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        tk.Button(grp_cg, text="DDA Line", command=lambda: self.set_draw_mode('dda')).pack(side=tk.LEFT, padx=1)
        tk.Button(grp_cg, text="Bresenham Line", command=lambda: self.set_draw_mode('bresenham')).pack(side=tk.LEFT, padx=1)
        tk.Button(grp_cg, text="Midpoint Circle", command=lambda: self.set_draw_mode('circle')).pack(side=tk.LEFT, padx=1)
        tk.Button(grp_cg, text="Draw Clipping Window", command=self.cmd_clipping_window, bg="#d62828", fg="white").pack(side=tk.LEFT, padx=1)
        tk.Button(grp_cg, text="Execute Cohen-Sutherland", command=self.cmd_cohen_sutherland, bg="#e76f51", fg="white").pack(side=tk.LEFT, padx=1)

        grp_trans = tk.LabelFrame(self.sub_container, text=" 2D Transformations & 3D ", bg="#2b2d42", fg="#ffb703", font=('Helvetica', 8, 'bold'), padx=5, pady=2)
        grp_trans.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        tk.Button(grp_trans, text="Rotate 90°", command=self.cmd_rotate).pack(side=tk.LEFT, padx=1)
        tk.Button(grp_trans, text="Scale 1.2x", command=self.cmd_scale, bg="#2a9d8f", fg="white").pack(side=tk.LEFT, padx=1)
        tk.Button(grp_trans, text="Horizontal Mirror", command=self.cmd_reflect).pack(side=tk.LEFT, padx=1)
        tk.Button(grp_trans, text="X-Shear", command=self.cmd_shear).pack(side=tk.LEFT, padx=1)
        tk.Button(grp_trans, text="Launch OpenGL 3D", command=self.cmd_opengl, bg="#ffb703", fg="black", font=('Helvetica', 8, 'bold')).pack(side=tk.LEFT, padx=2)

        grp_project = tk.LabelFrame(self.sub_container, text=" Lab 9 Medical Project ", bg="#2b2d42", fg="#e63946", font=('Helvetica', 8, 'bold'), padx=5, pady=2)
        grp_project.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        tk.Button(grp_project, text="Medical X-Ray Pipeline", command=self.cmd_xray_pipeline, bg="#e63946", fg="white", font=('Helvetica', 8, 'bold')).pack(side=tk.LEFT, padx=1)

        # --- Display Canvases ---
        self.canvas_frame = tk.Frame(self.root, bg="#1e1e24")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.lbl_orig = tk.Label(self.canvas_frame, text="Original Image View", bg="#111115", fg="#8d99ae")
        self.lbl_orig.grid(row=0, column=0, sticky="nsew", padx=5)

        self.lbl_proc = tk.Label(self.canvas_frame, text="Processed / Interactive Canvas View", bg="#111115", fg="#8d99ae")
        self.lbl_proc.grid(row=0, column=1, sticky="nsew", padx=5)

        self.lbl_proc.bind("<ButtonPress-1>", self.on_mouse_down)
        self.lbl_proc.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(1, weight=1)
        self.canvas_frame.grid_rowconfigure(0, weight=1)

        self.status = tk.Label(self.root, text="System Ready | Load an image or select an operation", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#2b2d42", fg="white")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp *.webp")])
        if file_path:
            self.root.config(cursor="watch")
            self.status.config(text="Loading image... Please wait.")
            self.root.update()

            self.original_img = cv2.imread(file_path)
            self.processed_img = self.original_img.copy()
            self.user_lines.clear()
            self.display_images()

            self.root.config(cursor="")
            self.status.config(text=f"Loaded Image: {file_path}")

    def save_image(self):
        if self.processed_img is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
            if file_path:
                cv2.imwrite(file_path, self.processed_img)

    def reset_image(self):
        if self.original_img is not None:
            self.processed_img = self.original_img.copy()
            self.draw_mode = None
            self.user_lines.clear()
            self.display_images()
            self.status.config(text="Reset to original canvas state")

    def display_images(self):
        if self.original_img is not None:
            orig_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            orig_pil = Image.fromarray(orig_rgb).resize((self.disp_w, self.disp_h))
            orig_tk = ImageTk.PhotoImage(orig_pil)
            self.lbl_orig.config(image=orig_tk)
            self.lbl_orig.image = orig_tk

        if self.processed_img is not None:
            proc_rgb = cv2.cvtColor(self.processed_img, cv2.COLOR_GRAY2RGB) if len(self.processed_img.shape) == 2 else cv2.cvtColor(self.processed_img, cv2.COLOR_BGR2RGB)
            proc_pil = Image.fromarray(proc_rgb).resize((self.disp_w, self.disp_h))
            proc_tk = ImageTk.PhotoImage(proc_pil)
            self.lbl_proc.config(image=proc_tk)
            self.lbl_proc.image = proc_tk

    def get_actual_coords(self, event):
        if self.processed_img is None:
            return 0, 0
        img_h, img_w = self.processed_img.shape[:2]
        scale_x = img_w / self.lbl_proc.winfo_width() if self.lbl_proc.winfo_width() > 1 else img_w / self.disp_w
        scale_y = img_h / self.lbl_proc.winfo_height() if self.lbl_proc.winfo_height() > 1 else img_h / self.disp_h
        return max(0, min(int(event.x * scale_x), img_w-1)), max(0, min(int(event.y * scale_y), img_h-1))

    def cmd_grayscale(self):
        self.processed_img = bf.to_grayscale(self.processed_img)
        self.display_images()
        self.status.config(text="Applied: Grayscale Conversion")

    def cmd_binary(self):
        self.processed_img = bf.to_binary(self.processed_img)
        self.display_images()
        self.status.config(text="Applied: Global Thresholding Segmentation")

    def cmd_hist_equalize(self):
        self.processed_img = bf.apply_histogram_equalization(self.processed_img)
        self.display_images()
        self.status.config(text="Applied: Histogram Equalization")

    def cmd_plot_hist(self):
        if not bf.plot_histogram(self.original_img, self.processed_img):
            messagebox.showwarning("Warning", "Please load an image first!")
        else:
            self.status.config(text="Displayed: Comparative Histogram Plots")

    def cmd_rgb_split(self):
        self.processed_img = bf.split_rgb_channels(self.processed_img)
        self.display_images()
        self.status.config(text="Applied: RGB Channel Decomposition")

    def cmd_depth_map(self):
        self.processed_img = bf.to_depth_map(self.processed_img)
        self.display_images()
        self.status.config(text="Applied: Pseudo 2D-to-3D Depth Map")

    def cmd_negative(self):
        self.processed_img = bf.to_negative(self.processed_img)
        self.display_images()
        self.status.config(text="Applied: Color Negative Inversion")

    def cmd_gaussian(self):
        self.processed_img = sf.apply_gaussian(self.processed_img)
        self.display_images()
        self.status.config(text="Applied: Gaussian Blur")

    def cmd_median(self):
        self.processed_img = sf.apply_median(self.processed_img)
        self.display_images()
        self.status.config(text="Applied: Median Filter")

    def cmd_sharpen(self):
        self.processed_img = sf.apply_sharpen(self.processed_img)
        self.display_images()
        self.status.config(text="Applied: Sharpening Kernel")

    def cmd_sobel(self):
        self.processed_img = sf.apply_sobel(self.processed_img)
        self.display_images()
        self.status.config(text="Applied: 1st Order Sobel Edge Detector")

    def cmd_laplacian(self):
        self.processed_img = sf.apply_laplacian(self.processed_img)
        self.display_images()
        self.status.config(text="Applied: 2nd Order Laplacian Edge Detector")

    def cmd_canny(self):
        self.processed_img = sf.apply_canny(self.processed_img)
        self.display_images()
        self.status.config(text="Applied: Canny Multi-Stage Edge Detector")

    def cmd_fft(self, filter_type):
        self.processed_img = sf.apply_fft_filter(self.processed_img, filter_type)
        self.display_images()
        self.status.config(text=f"Applied: 2D FFT {filter_type.upper()}-Pass Filter")

    def cmd_erosion(self):
        self.processed_img = morph.apply_erosion(self.processed_img)
        self.display_images()
        self.status.config(text="Applied: Morphological Erosion")

    def cmd_dilation(self):
        self.processed_img = morph.apply_dilation(self.processed_img)
        self.display_images()
        self.status.config(text="Applied: Morphological Dilation")

    def cmd_opening(self):
        self.processed_img = morph.apply_opening(self.processed_img)
        self.display_images()
        self.status.config(text="Applied: Morphological Opening")

    def cmd_closing(self):
        self.processed_img = morph.apply_closing(self.processed_img)
        self.display_images()
        self.status.config(text="Applied: Morphological Closing")

    def set_draw_mode(self, mode):
        self.draw_mode = mode
        self.status.config(text=f"Active Mode: {mode.upper()} Drawing. Click & Drag across Canvas.")

    def on_mouse_down(self, event):
        if self.draw_mode and self.processed_img is not None:
            self.start_x, self.start_y = self.get_actual_coords(event)

    def on_mouse_up(self, event):
        if self.draw_mode and self.start_x is not None and self.processed_img is not None:
            end_x, end_y = self.get_actual_coords(event)

            if self.draw_mode == 'dda':
                cv2.line(self.processed_img, (self.start_x, self.start_y), (end_x, end_y), (0, 165, 255), 2)
                self.user_lines.append((self.start_x, self.start_y, end_x, end_y))
            elif self.draw_mode == 'bresenham':
                cv2.line(self.processed_img, (self.start_x, self.start_y), (end_x, end_y), (0, 0, 255), 2)
                self.user_lines.append((self.start_x, self.start_y, end_x, end_y))
            elif self.draw_mode == 'circle':
                radius = int(np.sqrt((end_x - self.start_x)**2 + (end_y - self.start_y)**2))
                cv2.circle(self.processed_img, (self.start_x, self.start_y), radius, (0, 255, 0), 2)

            self.display_images()
            self.start_x = None

    def cmd_clipping_window(self):
        self.processed_img = g2d.draw_clipping_window(self.processed_img)
        self.display_images()
        self.status.config(text="Clipping Window Rendered!")

    def cmd_cohen_sutherland(self):
        if not self.user_lines:
            messagebox.showinfo("Instructions", "Draw at least one line segment using DDA or Bresenham before clipping!")
            return
        self.processed_img, count = g2d.execute_cohen_sutherland(self.original_img, self.user_lines)
        self.display_images()
        self.status.config(text=f"Cohen-Sutherland Completed: {count} line segment(s) clipped")

    def cmd_rotate(self):
        self.processed_img = t2d.rotate_image(self.processed_img)
        self.display_images()
        self.status.config(text="Rotated 90° Clockwise")

    def cmd_scale(self):
        self.processed_img = t2d.scale_image(self.processed_img)
        self.display_images()
        self.status.config(text="Scaled Up by 1.2x")

    def cmd_reflect(self):
        self.processed_img = t2d.reflect_image_horizontal(self.processed_img)
        self.display_images()
        self.status.config(text="Horizontal Reflection")

    def cmd_shear(self):
        self.processed_img = t2d.shear_image_x(self.processed_img)
        self.display_images()
        self.status.config(text="X-Axis Shearing Applied")

    def cmd_opengl(self):
        try:
            t2d.launch_opengl_window()
        except Exception as e:
            messagebox.showerror("Error", f"OpenGL Exception: {e}")

    def cmd_xray_pipeline(self):
        self.processed_img = med.run_xray_fracture_pipeline(self.processed_img)
        self.display_images()
        self.status.config(text="Medical Fracture Pipeline Applied!")
