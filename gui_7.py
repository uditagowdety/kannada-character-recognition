import tkinter as tk
from tkinter import Canvas, Button, Label, Frame, messagebox
import cv2
import numpy as np
from PIL import Image  # Import Pillow
import joblib

model=joblib.load("hog_model.pkl")

class_to_kannada = {
    1: 'ಅ', 2: 'ಆ', 3: 'ಇ', 4: 'ಈ', 5: 'ಉ', 6: 'ಊ', 7: 'ಋ', 8: 'ೠ', 
    9: 'ಎ', 10: 'ಏ', 11: 'ಐ', 12: 'ಒ', 13: 'ಓ', 14: 'ಔ', 15: 'ಅಂ', 16: 'ಅಃ', 
    17: 'ಕ', 18: 'ಖ', 19: 'ಗ', 20: 'ಘ', 21: 'ಙ', 
    22: 'ಚ', 23: 'ಛ', 24: 'ಜ', 25: 'ಝ', 26: 'ಞ', 
    27: 'ಟ', 28: 'ಠ', 29: 'ಡ', 30: 'ಢ', 31: 'ಣ', 
    32: 'ತ', 33: 'ಥ', 34: 'ದ', 35: 'ಧ', 36: 'ನ', 
    37: 'ಪ', 38: 'ಫ', 39: 'ಬ', 40: 'ಭ', 41: 'ಮ', 
    42: 'ಯ', 43: 'ರ', 44: 'ಲ', 45: 'ವ', 
    46: 'ಶ', 47: 'ಷ', 48: 'ಸ', 49: 'ಹ', 50: 'ಳ'
}

# Cardinal directions (up, right, down, left)
CARDINAL_DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

def find_starting_point(binary_image):
    """Find the first black pixel (object pixel) on the boundary of the binary image."""
    rows, cols = binary_image.shape
    for y in range(rows):
        for x in range(cols):
            if binary_image[y, x] == 0:  # 0 indicates a black pixel
                return x, y
    return None, None

def is_edge_pixel(binary_image, x, y):
    """Check if the pixel at (x, y) is an edge pixel."""
    # An edge pixel is defined as a black pixel (0) surrounded by at least one white pixel (255)
    if binary_image[y, x] != 0:
        return False
    
    rows, cols = binary_image.shape
    for dx, dy in CARDINAL_DIRECTIONS:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < cols and 0 <= new_y < rows:
            if binary_image[new_y, new_x] == 255:  # Check for white pixel
                return True
    return False

def get_crack_code(binary_image):
    """Trace the boundary of the object in the binary image using crack code."""
    x, y = find_starting_point(binary_image)
    if x is None or y is None:
        return []  # No boundary found

    crack_code = []
    visited = set()

    # Store the starting point
    start_x, start_y = x, y

    # Current direction of movement (start by moving up)
    direction_index = 0
    while True:
        visited.add((x, y))
        found_next = False

        # Try moving in the 4 cardinal directions
        for i in range(4):
            new_direction_index = (direction_index + i) % 4
            dx, dy = CARDINAL_DIRECTIONS[new_direction_index]
            new_x, new_y = x + dx, y + dy

            # Check bounds, object pixel, and whether it's an edge and not visited
            if (0 <= new_x < binary_image.shape[1] and
                0 <= new_y < binary_image.shape[0] and
                is_edge_pixel(binary_image, new_x, new_y) and
                (new_x, new_y) not in visited):

                # Store the direction (0=up, 1=right, 2=down, 3=left)
                crack_code.append(new_direction_index)
                x, y = new_x, new_y
                direction_index = new_direction_index
                found_next = True
                break

        if not found_next or (x == start_x and y == start_y):
            # No valid movement found or returned to starting point, boundary is closed
            break

    return crack_code

def compute_feature_vector(crack_code):
    """ Compute the feature vector for crack code directions """
    # Count occurrences of each direction (0 = Up, 1 = Right, 2 = Down, 3 = Left)
    counts = [0, 0, 0, 0]
    for direction in crack_code:
        counts[direction] += 1

    # Normalize the counts
    total_moves = len(crack_code) if len(crack_code) != 0 else 4
    normalized_histogram = [round(count / total_moves, 4) for count in counts]

    return counts, normalized_histogram

def preprocess_image(img, target_size=64):
    # Load the image
    # img = cv2.imread(image_path)
    
    # Invert the image colors
    img = 255 - img
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get binary image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Prepare an empty canvas for final output (just in case no contours are found)
    img_64 = np.full((target_size, target_size), 255, dtype=np.uint8)  # white background
    
    if contours:
        # Take the first contour
        cnt = contours[0]

        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)

        # Crop the image around the bounding box
        img_crop = img[y:y + h, x:x + w]

        # Get dimensions of the cropped image
        crop_h, crop_w = img_crop.shape[:2]

        # Calculate aspect ratio
        aspect_ratio = crop_w / crop_h

        # Desired size
        target_size = 64

        # Calculate new dimensions keeping the aspect ratio
        if aspect_ratio > 1:  # Wider than tall
            new_w = target_size
            new_h = int(target_size / aspect_ratio)
        else:  # Taller than wide
            new_h = target_size
            new_w = int(target_size * aspect_ratio)

        # Resize the cropped image to the new dimensions
        img_resized = cv2.resize(img_crop, (new_w, new_h))

        # Create a new image of size 512x512 with padding
        img_64 = np.full((target_size, target_size, 3), (0, 0, 0), dtype=np.uint8)  # black background

        # Calculate padding to center the resized image
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2

        # Place the resized image in the center of the 512x512 background
        img_64[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
        img_64 = cv2.cvtColor(img_64,cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(img_64, 127, 255, cv2.THRESH_BINARY)
        binary_image = cv2.bitwise_not(binary_image)
    
    else:
        print("No contours found.")
        binary_image = img_64
    
    return binary_image

def process_img(binary_image):
    """ Process the binary image to extract the feature vector """
    
    win_size = binary_image.shape
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9

    # Set the parameters of the HOG descriptor using the variables defined above
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    feature_vector = hog.compute(binary_image)

    return feature_vector.tolist()

class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kannada Character Recognition")
        self.root.geometry("900x600")
        self.root.configure(bg="lightgrey")

        # Set up main frames
        canvas_frame = Frame(root, bg="lightblue", bd=5, relief="ridge", width=400, height=600)
        canvas_frame.pack(side="left", padx=20, pady=20, fill="both", expand=True)
        canvas_frame.pack_propagate(False)  # Prevents resizing based on content

        control_frame = Frame(root, bg="lightgreen", bd=5, relief="ridge", width=400, height=600)
        control_frame.pack(side="right", padx=20, pady=20, fill="y")
        control_frame.pack_propagate(False)  # Prevents resizing based on content

        # Canvas for drawing
        self.canvas = Canvas(canvas_frame, width=400, height=400, bg="white", bd=3, relief="sunken")
        self.canvas.pack(padx=10, pady=10)

        # Buttons and prediction label
        self.process_button = Button(
            control_frame, text="Predict Character", command=self.process_image,
            height=2, width=20, font=("Helvetica", 16, "bold"), bg="blue", fg="white"
        )
        self.process_button.pack(pady=(20, 10))

        self.reset_button = Button(
            control_frame, text="Clear Canvas", command=self.reset_canvas,
            height=2, width=20, font=("Helvetica", 16, "bold"), bg="red", fg="white"
        )
        self.reset_button.pack(pady=10)

        self.prediction_label = Label(
            control_frame, text="Draw a character on the left and click Predict",
            font=("Helvetica", 18, "italic"), bg="lightgreen", wraplength=380, justify="center"
        )
        self.prediction_label.pack(pady=(30, 10), padx=10, anchor="n")

        # Bind drawing on canvas
        self.last_x, self.last_y = None, None
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line((self.last_x, self.last_y, x, y), fill="black", width=7)
        self.last_x = x
        self.last_y = y

    def reset(self, event):
        self.last_x, self.last_y = None, None
    
    def reset_canvas(self):
        """Clear the canvas and reset the prediction label."""
        self.canvas.delete("all")
        self.prediction_label.config(text="Draw a character and click Predict")

    def process_image(self):
        try:
            # Save canvas as an image
            self.canvas.update()
            self.canvas.postscript(file="drawing.eps")

            # Convert EPS to PNG using Pillow
            img = Image.open("drawing.eps")
            img.save("drawing.png", "png")

            # Read the PNG file with OpenCV
            img = cv2.imread("drawing.png")
            if img is None:
                raise ValueError("Could not read the image.")

            # Process image and predict
            binary_image = preprocess_image(img)
            feature_vector = process_img(binary_image)
            feature_vector = np.array(feature_vector).reshape(1, -1)
            prediction = model.predict(feature_vector)
            predicted_class = int(prediction[0])

            # Display prediction
            self.prediction_label.config(text=f"Predicted Character: {class_to_kannada[predicted_class]}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

# Run the application
root = tk.Tk()
app = DrawApp(root)
root.mainloop()

