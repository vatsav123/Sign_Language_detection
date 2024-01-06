import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("final_model.h5")  # Replace with the actual path to your saved model

# Create a function to match the labels to letter
def getLetter(result):
    classlabels = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
        19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X'
    }
    try:
        res = int(result)
        if res in classlabels:
            return classlabels[res]
        else:
            return "Label not found"
    except ValueError:
        return "Error"

class SignLanguageApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Sign Language Detection")

        self.label_result = tk.Label(self.master, text="Result:")
        self.label_result.pack(pady=10)

        self.canvas = tk.Canvas(self.master, width=300, height=300)
        self.canvas.pack(pady=10)

        self.btn_upload = tk.Button(self.master, text="Upload Image", command=self.upload_image)
        self.btn_upload.pack(pady=10)

        self.btn_predict = tk.Button(self.master, text="Predict", command=self.predict_image)
        self.btn_predict.pack(pady=10)

        self.img_path = None

    def upload_image(self):
        self.img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if self.img_path:
            self.show_image()

    def show_image(self):
        image = Image.open(self.img_path)
        image = image.resize((300, 300), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

    def predict_image(self):
        if self.img_path:
            image = Image.open(self.img_path).convert("L")  # Convert to grayscale
            image = image.resize((28, 28), Image.ANTIALIAS)
            image_array = np.array(image)
            image_array = image_array.reshape(1, 28, 28, 1)
            
            # Use the predict method instead of predict_classes
            predictions = model.predict(image_array)
            predicted_class = np.argmax(predictions)
            result = getLetter(predicted_class)
            self.label_result.config(text=f"Result: {result}")
        else:
            self.label_result.config(text="No image uploaded.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
