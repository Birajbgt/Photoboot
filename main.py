
import datetime
import cv2
import numpy as np

from app import *
# Load the overlay image (PNG with transparency)
overlay_img = cv2.imread('lastframe1.png', cv2.IMREAD_UNCHANGED)

# Extract the alpha channel from the overlay and create a mask
overlay_alpha = overlay_img[:, :, 3] / 255.0
overlay = overlay_img[:, :, :3]
background_alpha = 1.0 - overlay_alpha

# Resize overlay to desired dimensions if necessary
# For example, if you want the final video to be 640x480
desired_size = ( 1920   ,1080)
overlay = cv2.resize(overlay, desired_size)
overlay_alpha = cv2.resize(overlay_alpha, desired_size)
background_alpha = cv2.resize(background_alpha, desired_size)

 # Face detection
    num_faces = count_faces(resized_image)
    print(f"Number of faces: {num_faces}")  # Continuously print the number of faces
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Hand gesture recognition
    hand_results = hands.process(image_rgb)
    valid_gestures = 0
    current_gestures = []  # List to keep track of current hand gestures

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
cv2.namedWindow('Graduation', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Graduation', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Rendering")
countdown_active = False
last_valid_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to the same size as overlay
    frame = cv2.resize(frame, desired_size)
    for c in range(0, 3):
        frame[:, :, c] = (overlay_alpha * overlay[:, :, c] +
                          background_alpha * frame[:, :, c])

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
    cv2.setWindowProperty('Graduation', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if not countdown_active and valid_gestures >= num_faces and num_faces > 0:
        last_valid_time = datetime.datetime.now()
        countdown_active = True
    key = cv2.waitKey(1)
    if key != -1 and key != ord('q'):  # Any key is pressed, other than 'q'
        countdown_active = True
        last_valid_time = datetime.datetime.now()
    if countdown_active:
        elapsed = (datetime.datetime.now() - last_valid_time).total_seconds()
        if elapsed < 3:
            # Display countdown on the screen
            text_to_display = str(3 - int(elapsed))

            # Choose font for text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.5  # Increased font scale
            font_thickness = 3  # Adjust thickness as needed

            # Calculate text size
            text_size, _ = cv2.getTextSize(text_to_display, font, font_scale, font_thickness)

            # Calculate text position for center alignment
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2

            # Draw the text
            cv2.putText(frame, text_to_display, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
        else:
            # Capture snapshot and reset countdown
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # set_camera_resolution(cap, 2560, 1440)  # 2K resolution
            cv2.imshow('Graduation', frame)
            capture_snapshot(frame, f'snapshot_{timestamp}.png')
            print("Snapshot taken!")
            # set_camera_resolution(cap, 640, 480)
            countdown_active = False

    # Blend the overlay with the frame
    # bar_height = 50  # Adjust the height of the bar as needed
    # cv2.rectangle(frame, (0, frame.shape[0] - bar_height), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    # Display the resulting frame
    cv2.imshow('Graduation', frame)
    # Break the loop on pressing 'q'
    if key == ord('q'):  # Check for 'q' to exit
        break
    cv2.setWindowProperty('Graduation', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# Release everything when done
cap.release()
cv2.destroyAllWindows()
