# Label-Technical-Drawings-for-Object-Detection-Model
We are seeking skilled professionals to label technical drawings for our object detection model. The ideal candidate will have experience in machine learning and a strong background in accurately labeling visual data. Your attention to detail will ensure our model is trained effectively. If you are passionate about machine learning and have the relevant labeling experience, we want to hear from you!
===================
label technical drawings for an object detection model, you can use tools like LabelImg or CVAT (Computer Vision Annotation Tool). However, here's Python code to set up a basic custom labeling tool using Tkinter and integrate with machine learning pipelines.

This code lets users load images, draw bounding boxes around objects, and save labels in a format compatible with object detection models like YOLO or COCO.
1. Python Labeling Tool

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
import json

class LabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Technical Drawing Labeling Tool")

        # UI Elements
        self.canvas = tk.Canvas(root, width=800, height=600, bg="gray")
        self.canvas.pack(side=tk.LEFT)

        self.info_frame = tk.Frame(root)
        self.info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.load_button = tk.Button(self.info_frame, text="Load Images", command=self.load_images)
        self.load_button.pack(pady=10)

        self.save_button = tk.Button(self.info_frame, text="Save Labels", command=self.save_labels)
        self.save_button.pack(pady=10)

        self.class_entry = tk.Entry(self.info_frame, width=20)
        self.class_entry.pack(pady=10)
        self.class_entry.insert(0, "Enter Class")

        self.image_index = 0
        self.image_list = []
        self.current_image = None
        self.bboxes = []
        self.start_x, self.start_y = None, None

        # Canvas Bindings
        self.canvas.bind("<Button-1>", self.start_bbox)
        self.canvas.bind("<B1-Motion>", self.draw_bbox)
        self.canvas.bind("<ButtonRelease-1>", self.end_bbox)

    def load_images(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder:
            return
        self.image_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".png", ".jpg", ".jpeg"))]
        if not self.image_list:
            messagebox.showerror("Error", "No images found in the selected folder.")
            return
        self.image_index = 0
        self.load_image()

    def load_image(self):
        if self.image_index >= len(self.image_list):
            messagebox.showinfo("Done", "All images labeled!")
            return
        image_path = self.image_list[self.image_index]
        cv_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.current_image = ImageTk.PhotoImage(image=Image.fromarray(cv_image))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
        self.bboxes = []

    def save_labels(self):
        if not self.bboxes:
            messagebox.showinfo("Info", "No bounding boxes to save.")
            return
        labels_folder = "labels"
        os.makedirs(labels_folder, exist_ok=True)
        label_file = os.path.join(labels_folder, f"{os.path.basename(self.image_list[self.image_index])}.json")

        with open(label_file, "w") as f:
            json.dump({"image": self.image_list[self.image_index], "bboxes": self.bboxes}, f)

        self.image_index += 1
        self.load_image()

    def start_bbox(self, event):
        self.start_x, self.start_y = event.x, event.y

    def draw_bbox(self, event):
        self.canvas.delete("temp_bbox")
        self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="red", tag="temp_bbox")

    def end_bbox(self, event):
        end_x, end_y = event.x, event.y
        bbox_class = self.class_entry.get().strip()
        if not bbox_class:
            messagebox.showerror("Error", "Enter a valid class.")
            return
        self.bboxes.append({"x1": self.start_x, "y1": self.start_y, "x2": end_x, "y2": end_y, "class": bbox_class})
        self.canvas.create_rectangle(self.start_x, self.start_y, end_x, end_y, outline="green")
        self.canvas.create_text((self.start_x + end_x) // 2, (self.start_y + end_y) // 2, text=bbox_class, fill="blue")

# Run the tool
if __name__ == "__main__":
    root = tk.Tk()
    tool = LabelingTool(root)
    root.mainloop()

2. Features of the Tool

    Load Images: Load all images from a specified folder.
    Draw Bounding Boxes: Click and drag to draw bounding boxes on images.
    Class Labels: Enter a class label for each bounding box.
    Save Labels: Save labeled data in a JSON file for integration with models like YOLO or COCO.

3. Label Output Format

Each labeled image is saved as a JSON file with this structure:

{
  "image": "path/to/image.jpg",
  "bboxes": [
    {"x1": 50, "y1": 100, "x2": 200, "y2": 300, "class": "screw"},
    {"x1": 300, "y1": 150, "x2": 400, "y2": 250, "class": "bolt"}
  ]
}

4. Integration with Object Detection Models

    YOLO Format Conversion:
        Convert the JSON bounding boxes into YOLO text file format:

    <class_id> <x_center> <y_center> <width> <height>

COCO Format Conversion:

    Convert to COCO annotation format using Python:

    import json

    def convert_to_coco(json_files):
        coco_annotations = {"images": [], "annotations": [], "categories": []}
        # Populate the COCO structure
        return coco_annotations

5. Extending the Tool

    Predefined Classes: Use dropdown menus for common object classes.
    Image Navigation: Add "Next" and "Previous" buttons.
    Advanced Features: Use frameworks like LabelStudio or CVAT for more professional workflows.

This tool provides a great starting point for annotating technical drawings, ensuring high-quality labeled data for training your object detection model.
