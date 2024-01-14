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

    def process_frame(self, frame):
        # Resize frame for further processing
        width = int(frame.shape[1] * 0.5)
        height = int(frame.shape[0] * 0.5)
        resized_image = cv2.resize(frame, (width, height))
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Face detection
        num_faces = count_faces(resized_image)
        print(f"Number of faces: {num_faces}")  # Continuously print the number of faces
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Hand gesture recognition
        hand_results = hands.process(image_rgb)
        valid_gestures = 0
        current_gestures = []  # List to keep track of current hand gestures
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                gesture = classify_gesture(hand_landmarks.landmark)
                current_gestures.append(gesture)
                if gesture in ['Thumbs Up', 'Peace']:
                    valid_gestures += 1
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Check for valid shot and start countdown
        print(current_gestures)

        # Set window to fullscreen again after selection
        self.window.attributes("-fullscreen", True)

        if not self.countdown_active and valid_gestures >= num_faces and num_faces > 0:
            self.last_valid_time = datetime.datetime.now()
            self.countdown_active = True
            # Close the window when 'q' is pressed
        

        key = cv2.waitKey(1)
        if key != -1 and key != ord('q'):  # Any key is pressed, other than 'q'
            self.countdown_active = True
            self.last_valid_time = datetime.datetime.now()
        if self.countdown_active:
            elapsed = (datetime.datetime.now() - self.last_valid_time).total_seconds()
            
            if elapsed < 3:
                # Display countdown on the screen
                text_to_display = str(3 - int(elapsed))
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2.5
                font_thickness = 3
                text_size, _ = cv2.getTextSize(text_to_display, font, font_scale, font_thickness)
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = (frame.shape[0] + text_size[1]) // 2
                cv2.putText(frame, text_to_display, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
            else:
                # Capture snapshot and reset countdown
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                capture_snapshot(frame, f'snapshot_{timestamp}.png')
                print("Snapshot taken!")
                self.countdown_active = False

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
