import tkinter as tk

def on_button_click():
    value = entry.get()
    result_label.config(text=f"Hello, {value}!")

root = tk.Tk()
root.title("Photo Booth")

label = tk.Label(root, text="Enter your name:")
label.pack(pady=20)

entry = tk.Entry(root)
entry.pack(pady=20)

button = tk.Button(root, text="Greet Me", command=on_button_click)
button.pack(pady=20)

result_label = tk.Label(root, text="")
result_label.pack(pady=20)

root.mainloop()
