# Single Frame
import cv2
import mediapipe as mp
import datetime
import numpy as np
import requests
import qrcode
import io
import tempfile
import os
import time
import random
from PIL import Image, ImageDraw, ImageFont
from quotes import quotes

def upload_image_and_generate_qr(cv2_image):
    # Convert the OpenCV image to bytes
    _, im_buf_arr = cv2.imencode('.jpg', cv2_image)

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
        print(f"Image is available at: {image_url}")


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
    
    


def cv2_pil_to_cv2(pil_image):
    # Convert a PIL Image into an OpenCV image (in memory)
    open_cv_image = np.array(pil_image) 
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    return open_cv_image


def overlay_qr_code(cv2_image, qr_code_cv2):
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
    font_scale = 0.9
    font_thickness = 2
    text_x = x_offset  
    text_y = y_offset + size + 20  # 20 pixels below the QR code

    # Draw the text
    cv2.putText(cv2_image, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

    return cv2_image


# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands = 16)
face_detection = mp_face_detection.FaceDetection()

#for default frame
roi_x2, roi_y2, roi_width2, roi_height2 = 130, 480, 700, 700  # Example coordinates and size
#for second frame
roi_x, roi_y, roi_width, roi_height = 180, 180, 1600,1600  # Example coordinates and size
#for third frame
roi_x3, roi_y3, roi_width3, roi_height3 = 130, 480, 650, 700  # Example coordinates and size
#for fourth frame
roi_x4, roi_y4, roi_width4, roi_height4 = 130, 480, 650, 700  # Example coordinates and size


def is_thumb_extended(thumb_tip, thumb_mcp, other_fingers_mcp):
    # Check if the thumb tip is far from the MCP (metacarpophalangeal joint) of the thumb,
    # and closer to the MCPs of other fingers.
    return thumb_tip.y < thumb_mcp.y and all(thumb_tip.x < finger_mcp.x for finger_mcp in other_fingers_mcp)

def is_finger_folded(finger_tip, finger_mcp):
    # Check if the finger tip is close to its MCP.
    return finger_tip.y > finger_mcp.y

def resize_and_pad_image(image, max_height, side_padding, top_bottom_padding):
    h, w = image.shape[:2]
    scale = max_height / h
    resized_image = cv2.resize(image, (int(w * scale), max_height))

    # Add padding to the sides and top/bottom
    padded_image = cv2.copyMakeBorder(resized_image, top_bottom_padding, top_bottom_padding, 
                                      side_padding, side_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def set_camera_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def show_selection_window(image1,stream_window_name, image):
        # Initialize and display progress bar
    progress_bar_length = image.shape[1]
    progress_bar_height = 30
    progress_bar_bg_color = (0, 0, 0)  # Black background
    progress_bar_color = (0, 255, 0)  # Green progress

    def update_progress_bar(progress):
        progress_bar_img = np.zeros((progress_bar_height, progress_bar_length, 3), dtype=np.uint8)
        progress_bar_img[:] = progress_bar_bg_color
        progress_bar_img[:, :int(progress * progress_bar_length)] = progress_bar_color
        return progress_bar_img

    # Show progress bar during QR code generation
    progress = 0.0
    todate = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    f1i =  f"images/frame1_{todate}.png"
    cv2.imwrite(f1i, image1)

    max_height = 1000  # Increased max height for better quality
    gap_width = 20  # Width of the gap between images
    side_padding = 240  # Width of the side padding
    top_bottom_padding = 20  # Height of the top and bottom padding

    progress += 0.2
    progress_bar_img = update_progress_bar(progress)
    combined_with_progress = np.vstack((progress_bar_img, image))
    cv2.imshow(stream_window_name, combined_with_progress)
    cv2.waitKey(1)
    # Function to resize image with fixed height, maintaining aspect ratio, and add padding
    def resize_and_pad_image(image, max_height, side_padding, top_bottom_padding):
        h, w = image.shape[:2]
        scale = max_height / h
        resized_image = cv2.resize(image, (int(w * scale), max_height))

        # Add padding to the sides and top/bottom
        padded_image = cv2.copyMakeBorder(resized_image, top_bottom_padding, top_bottom_padding, 
                                          side_padding, side_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded_image

    # Resize and pad the images
    padded_image1 = resize_and_pad_image(image1, max_height, side_padding, top_bottom_padding)
    # padded_image2 = resize_and_pad_image(image2, max_height, side_padding, top_bottom_padding)
    progress += 0.5
    progress_bar_img = update_progress_bar(progress)
    combined_with_progress = np.vstack((progress_bar_img, image))
    cv2.imshow(stream_window_name, combined_with_progress)
    cv2.waitKey(1)
    # Calculate the new height after adding top and bottom padding
    new_height = max_height + 2 * top_bottom_padding

    # Create a gap (blank space) with the new height
    gap = np.zeros((new_height, gap_width, 3), dtype=np.uint8)

    # Combine the padded images with the gap in between
    # combined = np.hstack((padded_image1, gap, padded_image2))
    progress_bar_length = padded_image1.shape[1]

    while progress < 1.0:
        qr_code_image1 = upload_image_and_generate_qr(image1)
        progress += 0.5
        progress_bar_img = update_progress_bar(progress)
        combined_with_progress = np.vstack((progress_bar_img, padded_image1))
        cv2.imshow(stream_window_name, combined_with_progress)
        cv2.waitKey(1)

    while True:
        # Generate a new QR code for image1
        qr_code_image1 = upload_image_and_generate_qr(image1)
        image1_with_qr = overlay_qr_code(image1, qr_code_image1)

        # Resize and pad the image
        padded_image1 = resize_and_pad_image(image1_with_qr, max_height, side_padding, top_bottom_padding)

        # Display the image with the QR code
        cv2.imshow(stream_window_name, padded_image1)
        key = cv2.waitKey(1)

        if key != -1:  # Checks if any key is pressed
            break

    cv2.setWindowProperty(stream_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if key == ord('1'):
        return 1
    elif key == ord('2'):
        return 2
    else:
        return None

def classify_gesture(landmarks):
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
    if is_thumb_extended(thumb_tip, thumb_mcp, other_fingers_mcp) and \
       all(is_finger_folded(finger_tip, finger_mcp) for finger_tip, finger_mcp in 
           [(index_tip, index_mcp), (middle_tip, middle_mcp), (ring_tip, ring_mcp), (little_tip, little_mcp)]):
        return "Thumbs Up"
    

    # Peace Sign: Index and middle fingers extended, others folded
    if all(not is_finger_folded(finger_tip, finger_mcp) for finger_tip, finger_mcp in 
           [(index_tip, index_mcp), (middle_tip, middle_mcp)]) and \
       all(is_finger_folded(finger_tip, finger_mcp) for finger_tip, finger_mcp in 
           [(ring_tip, ring_mcp), (little_tip, little_mcp)]):
        return "Peace"

    return "Unknown"

def count_faces(image):
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Draw face detections
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)
        return len(results.detections)
    return 0
def cv2_pil_to_cv2(pil_image):
    # Convert a PIL Image into an OpenCV image (in memory)
    open_cv_image = np.array(pil_image) 
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    return open_cv_image
last_valid_time = None
countdown_active = False
overlay = cv2.imread('lastframe1.png', -1)
overlay_height, overlay_width = overlay.shape[:2]

def capture_snapshot(image, filename):

    #  # Resize overlay to match the frame size
    # resized_frame = cv2.resize(image, (roi_width, roi_height))
    # # Create a mask for the ROI based on the alpha channel of the overlay
    # roi_mask = overlay[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width, 3] / 255.0
    # roi_mask_inv = 1.0 - roi_mask
    # for c in range(0, 3):
    #         overlay[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width, c] = \
    #             resized_frame[:, :, c] * roi_mask_inv + overlay[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width, c] * roi_mask

    #     # Convert overlay to BGR (dropping alpha channel)
    # final_frame = overlay[:, :, :3]

    # Convert the OpenCV frame to a PIL image
    bar_height = 60  # Adjust the height of the bar as needed
    cv2.rectangle(image, (0, image.shape[0] - bar_height), (image.shape[1], image.shape[0]), (255, 255, 255), -1)

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
    cv2.putText(image, selected_quote, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    show_selection_window(image, 'Graduation', image)

    cv2.setWindowProperty('Graduation', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    # Clear the window
    # cv2.destroyAllWindows()
