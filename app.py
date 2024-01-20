import tkinter as tk
import cv2
from PIL import Image, ImageTk
import datetime
import mediapipe as mp
import random
from quotes import quotes
import tempfile
import requests
import qrcode
import numpy as np

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

        # video_frame_width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        # video_frame_height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        #..........................

        # Get screen dimensions for full screen
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        # Set video frame dimensions to full screen
        video_frame_width = screen_width
        video_frame_height = screen_height
        
        self.video_canvas = tk.Canvas(window, width=video_frame_width, height=video_frame_height)
        self.video_canvas.pack()


        self.snapshot_canvas = tk.Canvas(window, width=video_frame_width, height=video_frame_height)
        self.snapshot_canvas.pack_forget()
        # self.snapshot_canvas.place(in_=self.video_canvas)

        self.canvas = self.video_canvas
        self.video_canvas.bind("<Configure>", self.on_video_canvas_resize)  # Bind resize event for video canvas
        
        # self.btn_start = tk.Button(window, text="Start", width=10, command=self.start_capture)
        # self.btn_start.pack(padx=10, pady=5)
        
        # self.btn_stop = tk.Button(window, text="Stop", width=10, command=self.stop_capture)
        # self.btn_stop.pack(padx=10, pady=5)

        self.mp_hands = mp.solutions.hands
        self.mp_face_detection = mp.solutions.face_detection
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=16)
        self.face_detection = self.mp_face_detection.FaceDetection()
        self.mp_drawing = mp.solutions.drawing_utils

        self.selection_frame = tk.Frame(window)  # Frame to display selection
        self.selection_frame.pack(fill=tk.BOTH, expand=True)

        self.selection_label = tk.Label(self.selection_frame)
        self.selection_label.pack(fill=tk.BOTH, expand=True)


        self.small_images_frame = tk.Frame(window)
        self.small_images_frame.pack(side=tk.BOTTOM, pady=10)

        small_image_paths = ['lastframe1.png', 'lastframe2.png', 'lastframe3.png','lastframe4.png']
        self.small_images = [tk.PhotoImage(file=path) for path in small_image_paths]

        # Create and display the small image canvases
        self.small_image_canvases = []

        for index, img in enumerate(self.small_images):
            small_image_canvas = tk.Canvas(self.small_images_frame, width=50, height=50, bg="white")
            small_image_canvas.create_image(0, 0, anchor=tk.NW, image=img)
            small_image_canvas.bind('<Button-1>', lambda event, index=index: self.on_small_image_click(index))
            self.small_image_canvases.append(small_image_canvas)

        # Pack the small image canvases side by side
        for canvas in self.small_image_canvases:
            canvas.pack(side=tk.LEFT, padx=10)

        
        self.update()
        self.window.bind('<KeyPress>', self.on_key_press)  # Bind key press event once
        self.window.mainloop()

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

# Remove landmarks in pics
                # self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        
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

  
    def show_selection_window(self, image):
        print("video before selection" if self.canvas == self.video_canvas else "current before selection")
        self.video_canvas.pack_forget()  # Hide video canvas
        print("video after pack.forget video" if self.canvas == self.video_canvas else "current after pack.forget selection")
        self.snapshot_canvas.pack()       # Display snapshot canvas
        print("video after snapshott pack" if self.canvas == self.video_canvas else "current after snapshot pack")
        self.canvas = self.snapshot_canvas 
        print("video after snapshott pack" if self.canvas == self.video_canvas else "current after snapshot pack")

        # Clear previous content in the selection frame

        # Generate a new QR code for the captured image
        qr_code_image = self.upload_image_and_generate_qr(image)

        # Overlay the QR code on the captured image
        image_with_qr = self.overlay_qr_code(image, qr_code_image)

        # Resize and pad the image for better visibility
        max_height = 800
        side_padding = 50
        top_bottom_padding = 20
        padded_image = self.resize_and_pad_image(image_with_qr, max_height, side_padding, top_bottom_padding)

        # Convert the OpenCV image to RGB format for displaying with Tkinter
        image_rgb = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))

        # Create a label in the selection frame to display the image
        label = tk.Label(self.selection_frame, image=photo)
        label.image = photo  # Keep a reference to the image to prevent garbage collection
        label.pack(fill=tk.BOTH, expand=True)

        # Update the label in the selection frame
        self.selection_label = label




    def upload_image_and_generate_qr(self, cv2_image):
            # Convert the OpenCV image to bytes
            _, im_buf_arr = cv2.imencode('.jpg', cv2_image)

            # Save the image to a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_file.write(im_buf_arr)
            temp_file.close()

            # Upload the image file
            with open(temp_file.name, 'rb') as file:
                files = {'file': file}
                response = requests.post('https://file.io', files=files)
            
            # Check for successful upload
            if response.status_code == 200 and response.json()['success']:
                # Get the URL of the uploaded image
                image_url = response.json()['link']
                print(image_url)

                # Generate QR Code for the URL
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_L,
                    box_size=15,
                    border=1,
                )
                qr.add_data(image_url)
                qr.make(fit=True)
                qr_img = qr.make_image(fill='black', back_color='white')
                qr_img = qr_img.convert('RGB')

                # Convert PIL Image to OpenCV Image
                qr_open_cv_image = np.array(qr_img) 
                qr_open_cv_image = qr_open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
                return qr_open_cv_image
            else:
                raise Exception("Error uploading file")

    def overlay_qr_code(self, cv2_image, qr_code_cv2):
        # Size of the QR code
        size = 200  # Increased size for better visibility
        qr_code_resized = cv2.resize(qr_code_cv2, (size, size))
        
        # Position for QR code overlay (bottom-right corner)
        x_offset = cv2_image.shape[1] - size - 20  # 20 pixels from the right edge
        y_offset = 0

        # Overlay QR code
        cv2_image[y_offset:y_offset + size, x_offset:x_offset + size] = qr_code_resized

        # Position for text
        text = "Scan to get Photo"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4.0  # Increase the font scale
        font_thickness = 2
        text_x = x_offset  
        text_y = y_offset + size + 20  # 20 pixels below the QR code

        # Draw the text
        cv2.putText(cv2_image, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

        return cv2_image

    def resize_and_pad_image(self, image, max_height, side_padding, top_bottom_padding):
        h, w = image.shape[:2]

        # Calculate the scaling factor to fit within the specified max height
        scale = max_height / h

        # Resize the image while maintaining the aspect ratio
        resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))

        # Calculate the new height and width after resizing
        new_h, new_w = resized_image.shape[:2]

        # Calculate the padding needed
        pad_top = (max_height - new_h) // 2
        pad_bottom = max_height - new_h - pad_top

        # Add padding to the sides and top/bottom
        padded_image = cv2.copyMakeBorder(resized_image, pad_top + top_bottom_padding, pad_bottom + top_bottom_padding, side_padding, side_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return padded_image

    def on_small_image_click(self, index):
        # Handle the click event for the small image
        print(f"Small Image {index + 1} Clicked")

        # Update the selected small image index
        self.selected_small_image_index = index

        # Update the border image with the clicked image
        clicked_image_path = f'lastframe{index + 1}.png'
        
        if index == 4:
            border_image = cv2.imread(clicked_image_path, cv2.IMREAD_UNCHANGED)
            border_image = cv2.resize(border_image, (self.video_canvas.winfo_reqwidth(), self.video_canvas.winfo_reqheight()))
        else:
            border_image = cv2.imread(clicked_image_path, cv2.IMREAD_UNCHANGED)
            border_image = cv2.resize(border_image, (self.video_canvas.winfo_reqwidth(), self.video_canvas.winfo_reqheight()))

        # Update the small image canvases to reflect the updated border image
        for small_image_canvas in self.small_image_canvases:
            small_image_canvas.delete("all")
            small_image_canvas.create_image(0, 0, anchor=tk.NW, image=tk.PhotoImage(file=clicked_image_path))
            small_image_canvas.image = tk.PhotoImage(file=clicked_image_path)

        # Update the border image for the snapshot capture
        self.border_image = border_image

    def on_video_canvas_resize(self, event):
        # Handle resize event for video canvas
        canvas_width = event.width
        canvas_height = event.height

        # Calculate the total width occupied by small image canvases
        total_small_images_width = (50 + 10) * len(self.small_image_canvases)

        # Calculate the x-coordinate to place the small images at the bottom center
        x_coordinate = (canvas_width - total_small_images_width) // 2

        # Place the small images frame at the bottom center
        self.small_images_frame.place(x=x_coordinate, y=canvas_height - 60)

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = VideoCaptureApp(root, "Video Capture App")

#...........................................
if __name__ == "__main__":
    root = tk.Tk()

    # Maximize the Tkinter window
    root.attributes('-fullscreen', True)
    root.bind('<Escape>', lambda event: root.attributes('-fullscreen', False))

    app = VideoCaptureApp(root, "Video Capture App")

