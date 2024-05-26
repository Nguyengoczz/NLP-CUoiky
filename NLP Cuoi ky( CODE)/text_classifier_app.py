import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext
from tkinter import filedialog
from text_classifier import classify_text

class TextClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Text Classifier")
        self.geometry("600x400")
        
        self.create_widgets()

    def create_widgets(self):
        self.text_box = scrolledtext.ScrolledText(self, width=60, height=10, wrap=tk.WORD)
        self.text_box.pack(pady=10)

        self.classify_button = tk.Button(self, text="Classify Text", command=self.classify_text)
        self.classify_button.pack(pady=10)

        self.clear_button = tk.Button(self, text="Clear", command=self.clear_text)
        self.clear_button.pack(pady=10)

        self.load_model_button = tk.Button(self, text="Load Model", command=self.load_model)
        self.load_model_button.pack(pady=10)

    def classify_text(self):
        text = self.text_box.get("1.0", tk.END)
        if text.strip():
            result = classify_text(text.strip())
            messagebox.showinfo("Classification Result", f"The text belongs to category: {result}")
        else:
            messagebox.showwarning("Empty Text", "Please enter some text to classify.")

    def clear_text(self):
        self.text_box.delete("1.0", tk.END)

    def load_model(self):
        model_file = filedialog.askopenfilename(initialdir="./", title="Select Model File",
                                                 filetypes=(("Pickle files", "*.pkl"), ("All files", "*.*")))
        if model_file:
            messagebox.showinfo("Model Loaded", f"Model loaded successfully from {model_file}")
            # Load model code here
        else:
            messagebox.showwarning("No Model File", "Please select a model file.")

if __name__ == "__main__":
    app = TextClassifierApp()
    app.mainloop()
