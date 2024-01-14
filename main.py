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
    def process_frame(self, frame):
        # Resize frame for further processing
        width = int(frame.shape[1] * 0.5)
        height = int(frame.shape[0] * 0.5)
        resized_image = cv2.resize(frame, (width, height))
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Face detection
        num_faces = self.count_faces(resized_image)
        print(f"Number of faces: {num_faces}")  # Continuously print the number of faces
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Hand gesture recognition
        hand_results = self.hands.process(image_rgb)
        valid_gestures = 0
        current_gestures = []  # List to keep track of current hand gestures
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                gesture = self.classify_gesture(hand_landmarks.landmark)
                current_gestures.append(gesture)
                if gesture in ['Thumbs Up', 'Peace']:
                    valid_gestures += 1
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        
        # Check for valid shot and start countdown
        print(current_gestures)
        elapsed = (datetime.datetime.now() - self.last_valid_time).total_seconds()


        if not self.countdown_active and valid_gestures >= num_faces and num_faces > 0 and not self.snapshot_taken:
            self.last_valid_time = datetime.datetime.now()
            self.countdown_active = True


        if self.countdown_active:
            
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
            elif elapsed >= 3 and not self.snapshot_taken:
                # Capture snapshot and reset countdown
                self.countdown_active = False  # Reset the countdown
                self.snapshot_taken = True
                self.capture_snapshot(frame)

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

        self.window.after(10, self.update)


    def on_key_press(self, event):
        if event.char.lower() == 'q':
            print("video" if self.canvas == self.video_canvas else "current")
            if self.canvas == self.video_canvas:
                # If currently in video capture, quit the application
                self.window.destroy()

            elif self.canvas == self.snapshot_canvas:
                # If currently in snapshot capture, switch to video capture
                self.snapshot_canvas.pack_forget()  # Hide the snapshot canvas
                self.video_canvas.pack()   # Display the video canvas
                print("Before switch")
                print("Video Canvas " if self.canvas == self.video_canvas else "Snapshot Canvas")
                print("After switch")
                self.canvas = self.video_canvas 
                print("Video Canvas " if self.canvas == self.video_canvas else "Snapshot Canvas")
                self.snapshot_taken = False
                self.selection_label.destroy()


        else:
            self.countdown_active = True
            self.last_valid_time = datetime.datetime.now()
            print("*"*50)

    def convert_frame_to_photo(self, frame):
        # Resize the frame to match the window dimensions
        width = int(self.canvas.winfo_reqwidth())
        height = int(self.canvas.winfo_reqheight())
        frame = cv2.resize(frame, (width, height))

        # Convert frame to RGB and create a PhotoImage
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        return photo

    def count_faces(self, image):
        # Convert the image to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run face detection on the RGB image
        results = self.face_detection.process(image_rgb)

        # Draw face detections
        if results.detections:
            for detection in results.detections:
                self.mp_drawing.draw_detection(image, detection)
            return len(results.detections)
        return 0

    def classify_gesture(self, landmarks):
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        index_tip = landmarks[8]
        index_mcp = landmarks[5]
        middle_tip = landmarks[12]
        middle_mcp = landmarks[9]
        ring_tip = landmarks[16]
        ring_mcp = landmarks[13]
        little_tip = landmarks[20]
        little_mcp = landmarks[17]

        other_fingers_mcp = [index_mcp, middle_mcp, ring_mcp, little_mcp]

        # Thumbs Up: Thumb extended, other fingers folded
        if self.is_thumb_extended(thumb_tip, thumb_mcp, other_fingers_mcp) and \
           all(self.is_finger_folded(finger_tip, finger_mcp) for finger_tip, finger_mcp in 
               [(index_tip, index_mcp), (middle_tip, middle_mcp), (ring_tip, ring_mcp), (little_tip, little_mcp)]):
            return "Thumbs Up"

        # Peace Sign: Index and middle fingers extended, others folded
        if all(not self.is_finger_folded(finger_tip, finger_mcp) for finger_tip, finger_mcp in 
               [(index_tip, index_mcp), (middle_tip, middle_mcp)]) and \
           all(self.is_finger_folded(finger_tip, finger_mcp) for finger_tip, finger_mcp in 
               [(ring_tip, ring_mcp), (little_tip, little_mcp)]):
            return "Peace"

        return "Unknown"

    def is_thumb_extended(self, thumb_tip, thumb_mcp, other_fingers_mcp):
        # Check if the thumb tip is far from the MCP (metacarpophalangeal joint) of the thumb,
        # and closer to the MCPs of other fingers.
        return thumb_tip.y < thumb_mcp.y and all(thumb_tip.x < finger_mcp.x for finger_mcp in other_fingers_mcp)

    def is_finger_folded(self, finger_tip, finger_mcp):
        # Check if the finger tip is close to its MCP.
        return finger_tip.y > finger_mcp.y

    def capture_snapshot(self, image):
        # bar_height = 60  # Adjust the height of the bar as needed
        # cv2.rectangle(image, (0, image.shape[0] - bar_height), (image.shape[1], image.shape[0]), (255, 255, 255), -1)


        selected_quote = random.choice(quotes)

        # Choose font for text
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        font_scale = 1.5  # Adjust font scale as needed
        font_thickness = 3  # Adjust thickness as needed

        # Calculate text size
        text_size, _ = cv2.getTextSize(selected_quote, font, font_scale, font_thickness)

        # Calculate text position for center alignment
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = image.shape[0] - 20  # Position from the bottom

        # Draw the text
        image_with_border = self.overlay_border(image, self.border_image)

        cv2.putText(image_with_border, selected_quote, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

        self.show_selection_window(image_with_border)


    def overlay_border(self, cv2_image, border_image):
        # Extract the alpha channel from the PNG image and normalize to the range [0, 1]
        alpha_channel = border_image[:, :, 3] / 255.0

        # Resize the alpha channel to match the size of the captured frame
        alpha_channel = cv2.resize(alpha_channel, (cv2_image.shape[1], cv2_image.shape[0]))

        # Extract the RGB channels from the PNG image
        rgb_channels = border_image[:, :, :3]

        # Resize the RGB channels to match the size of the captured frame
        rgb_channels = cv2.resize(rgb_channels, (cv2_image.shape[1], cv2_image.shape[0]))

        # Ensure alpha_channel is a 2D array to avoid broadcasting issues
        alpha_channel = alpha_channel[:, :, np.newaxis]

        # Combine the captured frame and the PNG image with proper transparency
        combined_image = (1.0 - alpha_channel) * cv2_image + alpha_channel * rgb_channels

        return combined_image.astype(np.uint8)


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoCaptureApp(root, "Video Capture App")
