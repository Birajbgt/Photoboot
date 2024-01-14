import tkinter as tk
import cv2
from PIL import Image, ImageTk
import datetime
from app import count_faces, classify_gesture, hands, mp_drawing, mp_hands, capture_snapshot

class VideoCaptureApp:
    def __init__(self, window, window_title, video_source=0):

        self.border_image = None

        self.snapshot_taken = False 
        self.countdown_active = False
        self.last_valid_time = datetime.datetime.now()

        self.window = window
        self.window.title(window_title)
        
        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)

        video_frame_width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_frame_height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        self.video_canvas = tk.Canvas(window, width=video_frame_width, height=video_frame_height)
        self.video_canvas.pack()
        
        self.btn_stop = tk.Button(window, text="Stop", width=10, command=self.stop_capture)
        self.btn_stop.pack(padx=10, pady=5)
        
        self.update()
        self.window.mainloop()

    
    def start_capture(self):
        self.vid = cv2.VideoCapture(self.video_source)

    def stop_capture(self):
        self.vid.release()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.photo = self.convert_frame_to_photo(frame)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.process_frame(frame)
        self.window.bind('<KeyPress-q>', self.on_key_press)

        self.window.after(10, self.update)

    def on_key_press(self, event):
        if event.char.lower() == 'q':
            self.window.destroy()

        else:
            self.countdown_active = True
            self.last_valid_time = datetime.datetime.now()

    def convert_frame_to_photo(self, frame):
        # Resize the frame to match the window dimensions
        width = int(self.canvas.winfo_reqwidth())
        height = int(self.canvas.winfo_reqheight())
        frame = cv2.resize(frame, (width, height))

        # Convert frame to RGB and create a PhotoImage
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        return photo

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoCaptureApp(root, "Video Capture App")
