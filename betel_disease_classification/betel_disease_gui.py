import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="model/model_unquant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Function to classify an image
def classify_image(image_path):
    # Load image and resize it to the expected input shape
    image = cv2.imread(image_path)
    image = cv2.resize(image, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32) / 255.0

    # Set the tensor to the input image
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Get the result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    result = np.squeeze(output_data)

    # Find the label with the highest probability
    top_result = np.argmax(result)
    return labels[top_result], result[top_result]

# Function to open a file dialog and classify the selected image
def open_and_classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Display the image
        image = Image.open(file_path)
        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
        
        # Classify the image
        label, confidence = classify_image(file_path)
        result_label.config(text=f"Prediction: {label} with confidence {confidence:.2f}")

# Create the main window
root = tk.Tk()
root.title("Betel Disease Classification")

# Create a button to open the file dialog
browse_button = tk.Button(root, text="Browse Image", command=open_and_classify_image)
browse_button.pack()

# Create a label to display the image
image_label = tk.Label(root)
image_label.pack()

# Create a label to display the result
result_label = tk.Label(root, text="")
result_label.pack()

# Run the Tkinter event loop
root.mainloop()
