import cv2
import torch
import numpy as np
import time
import os
from pathlib import Path
import sys

# --- Configuration Parameters ---
# Default thresholds (0.0 to 1.0) for confidence and IoU
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
# Input image size for the YOLOv5 model (usually 640x640)
IMG_SIZE = 640 
# Path to your cloned YOLOv5 repository. 
# Make sure 'yolov5' folder is in the same directory as this script.
YOLOV5_REPO_PATH = Path("yolov5") 

# --- Setup YOLOv5 Path and Imports ---
# Add the YOLOv5 directory to the Python path to allow importing its modules
if not YOLOV5_REPO_PATH.is_dir():
    print(f"Error: YOLOv5 directory not found at '{YOLOV5_REPO_PATH}'.")
    print("Please ensure the 'yolov5' folder is in the same directory as this script,")
    print("or update the 'YOLOV5_REPO_PATH' variable to point to your YOLOv5 clone.")
    sys.exit(1)

if str(YOLOV5_REPO_PATH) not in sys.path:
    sys.path.append(str(YOLOV5_REPO_PATH))

try:
    # Import necessary components from yolov5
    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_boxes
    from utils.torch_utils import select_device
    from utils.plots import Annotator, colors # For drawing bounding boxes and labels
    
except ImportError as e:
    print(f"\nError importing YOLOv5 modules: {e}")
    print("Please ensure the 'yolov5' folder is correctly configured in sys.path.")
    print(f"Also, run 'pip install -r {YOLOV5_REPO_PATH}/requirements.txt' to install all dependencies.")
    sys.exit(1)

# --- Load YOLOv5 Model ---
# Path to the pre-trained YOLOv5 weights (e.g., yolov5s.pt, yolov5m.pt, etc.)
# This file should be inside your YOLOV5_REPO_PATH folder.
weights_path = YOLOV5_REPO_PATH / 'yolov5s.pt' 
if not weights_path.is_file():
    print(f"\nError: YOLOv5 weights file not found at '{weights_path}'.")
    print("Please ensure 'yolov5s.pt' exists in your yolov5 folder.")
    print("You might need to download it or run `python detect.py --source 0` once from the yolov5 directory.")
    sys.exit(1)

# Select device (CPU or GPU). '' for auto-select (prefers GPU if available).
device = select_device('') 

try:
    # Load the full model architecture and weights
    print(f"Attempting to load YOLOv5 model from '{weights_path}' on {device}...")
    model = attempt_load(weights_path, device=device, inplace=True, fuse=True) 
    model.eval() # Set model to evaluation mode
    stride = int(model.stride.max()) # Model stride, typically 32
    names = model.module.names if hasattr(model, 'module') else model.names # Get class names

    print(f"Model loaded successfully. Found {len(names)} classes: {names}")
except Exception as e:
    print(f"\nError loading YOLOv5 model from '{weights_path}': {e}")
    print("Possible reasons: Corrupted weights file, incompatible PyTorch version, or CUDA issues.")
    sys.exit(1)

# --- Global Variables for Features ---
cap = None # OpenCV VideoCapture object
webcam_active = False # Flag to track webcam status
detection_paused = False # Flag to pause/resume detection processing
snapshot_count = 0 # Counter for saved snapshots

# FPS Calculation
fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0

# Thresholds for trackbars (converted to 0-100 integer range)
conf_threshold = int(DEFAULT_CONF_THRESHOLD * 100) 
iou_threshold = int(DEFAULT_IOU_THRESHOLD * 100) 

# Class Filtering: A dictionary to store state (True/False) for each class ID
# Initially, all classes are enabled for detection
class_filter_state = {i: True for i in range(len(names))}
# List of class IDs that are currently enabled for filtering (used by non_max_suppression)
filtered_classes = [cid for cid, state in class_filter_state.items() if state]

# Save Detections to File
save_detections_to_file = False
output_txt_dir = Path("runs/detect_logs")
output_txt_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

# --- OpenCV Trackbar Callbacks ---
# These functions are called when the trackbar position changes
def on_conf_trackbar(val):
    global conf_threshold
    conf_threshold = val

def on_iou_trackbar(val):
    global iou_threshold
    iou_threshold = val

# Function to toggle class filter state (called by keyboard input)
def toggle_class_filter(class_id):
    global filtered_classes, class_filter_state
    if 0 <= class_id < len(names):
        class_filter_state[class_id] = not class_filter_state[class_id] # Toggle the state
        # Rebuild the list of active class IDs based on the new state
        filtered_classes = [cid for cid, state in class_filter_state.items() if state]
        
        active_class_names = [names[c] for c in filtered_classes] if filtered_classes else ['ALL']
        print(f"Toggled '{names[class_id]}'. Current active classes: {', '.join(active_class_names)}")
    else:
        print(f"Warning: Invalid class ID {class_id}. Model has only {len(names)} classes.")

# --- Main Application Loop ---
print("\n--- Controls ---")
print(" 's' : Start Webcam")
print(" 'q' / 'ESC' : Quit Application")
print(" 'c' : Capture Snapshot (saves image to current directory)")
print(" 'p' : Toggle Pause/Resume Object Detection")
print(" 'f' : Toggle Saving Detections to TXT file (in runs/detect_logs/)")
print(" '0'-'9' : Toggle detection for specific classes (maps to first 10 classes)")
print("-----------------\n")

# Create a resizable OpenCV window
cv2.namedWindow("YOLOv5 Detection", cv2.WINDOW_NORMAL) 

# Create trackbars for real-time threshold adjustment
cv2.createTrackbar("Conf_Threshold", "YOLOv5 Detection", conf_threshold, 100, on_conf_trackbar)
cv2.createTrackbar("IoU_Threshold", "YOLOv5 Detection", iou_threshold, 100, on_iou_trackbar)

while True:
    # Get current threshold values from trackbars (converted to float for model)
    current_conf_thresh_float = conf_threshold / 100.0
    current_iou_thresh_float = iou_threshold / 100.0

    # Wait for a key press (1ms delay), and get its ASCII value
    key = cv2.waitKey(1) & 0xFF

    # --- Handle Key Presses ---
    if key == ord('s') and not webcam_active:
        cap = cv2.VideoCapture(0) # Open default webcam (index 0)
        if not cap.isOpened():
            print("Error: Cannot access webcam. Please check if it's connected and not in use by another application.")
            webcam_active = False # Keep webcam_active False if opening fails
            cap = None # Reset cap object
            # Display a warning in the window
            temp_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(temp_frame, "WEBCAM ERROR! Check console.", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("YOLOv5 Detection", temp_frame)
            continue # Continue loop, waiting for 's' again
        else:
            print("Webcam started. Press 'p' to pause detection.")
            webcam_active = True
            fps_start_time = time.time() # Reset FPS counter on start
            fps_frame_count = 0
            detection_paused = False # Ensure detection is not paused when starting

    elif key == ord('q') or key == 27: # 'q' or 'ESC' key to quit
        if cap:
            cap.release() # Release webcam resource
        cv2.destroyAllWindows() # Close all OpenCV windows
        print("Webcam stopped and program exited.")
        break # Exit the main loop

    elif key == ord('c') and webcam_active:
        if cap and cap.isOpened(): # Ensure webcam is active and open
            ret, frame = cap.read()
            if ret:
                timestamp = int(time.time()) # Use timestamp for unique filename
                filename = f"snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame) # Save the current frame
                print(f"Snapshot saved: {filename}")
                snapshot_count += 1
            else:
                print("Failed to capture snapshot: Could not read frame from webcam.")
        else:
            print("Cannot capture snapshot: Webcam is not active.")

    elif key == ord('p'): # Toggle pause/resume detection
        detection_paused = not detection_paused
        print(f"Detection {'PAUSED' if detection_paused else 'RESUMED'}")
    
    elif key == ord('f'): # Toggle saving detections to file
        save_detections_to_file = not save_detections_to_file
        print(f"Saving detections to file: {'ENABLED' if save_detections_to_file else 'DISABLED'}")

    elif ord('0') <= key <= ord('9'): # Toggle specific class detection (keys 0-9)
        class_id_to_toggle = key - ord('0')
        toggle_class_filter(class_id_to_toggle)

    # --- Webcam Frame Processing and Detection ---
    if webcam_active and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam. Reattempting to get frame...")
            # If a frame can't be read, assume webcam issue and try to re-initialize on next 's'
            webcam_active = False 
            if cap: cap.release()
            cap = None
            continue # Skip to next loop iteration

        # Create an Annotator object for drawing bounding boxes and labels
        # It takes the frame as input and uses YOLOv5's plotting utilities
        annotator = Annotator(frame, line_width=2, example=str(names))

        detections_in_frame = 0 # Reset object count for current frame
        detection_log_data = [] # List to store detection info for file logging

        if not detection_paused:
            # --- Prepare image for model inference ---
            img_h, img_w, _ = frame.shape # Get original frame dimensions
            
            # Convert BGR (OpenCV default) to RGB (YOLOv5 expects RGB)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize image to model's expected input size (e.g., 640x640)
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Convert numpy array to PyTorch tensor, transpose to (C, H, W), normalize to 0-1
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            # Add batch dimension (B, C, H, W) and move to appropriate device (CPU/GPU)
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # --- Perform inference ---
            with torch.no_grad(): # Disable gradient calculation for inference (saves memory/time)
                pred = model(img_tensor)[0] # Get raw predictions from the model
                
                # Apply Non-Max Suppression (NMS) to filter redundant detections
                # 'classes' argument allows filtering by specific class IDs
                pred = non_max_suppression(pred, 
                                           conf_thres=current_conf_thresh_float, 
                                           iou_thres=current_iou_thresh_float, 
                                           classes=filtered_classes if filtered_classes else None)
                                           # If filtered_classes is empty, it detects all classes (None)

            # --- Process Detections ---
            if pred and len(pred) > 0: # Check if there are any detections in the batch
                det = pred[0] # Get detections for the first (and only) image in the batch

                if len(det): # If there are actual detections
                    # Rescale bounding box coordinates from IMG_SIZE to original frame dimensions
                    det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()

                    # Iterate through each detected object
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}' # Format label: "ClassName Confidence"
                        # Draw bounding box and label using the Annotator
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))
                        detections_in_frame += 1 # Increment object count

                        # If saving detections to file is enabled, prepare data
                        if save_detections_to_file:
                            x1, y1, x2, y2 = map(int, xyxy)
                            detection_log_data.append(f"{names[int(cls)]},{conf:.4f},{x1},{y1},{x2},{y2}\n")
                    
                    # Write all detections for the current frame to a log file
                    if save_detections_to_file and detection_log_data:
                        log_filename = output_txt_dir / f"detections_{int(time.time())}.txt"
                        with open(log_filename, 'a') as f: # 'a' for append mode
                            f.writelines(detection_log_data)
                        # print(f"Detections logged to {log_filename}") # Uncomment for verbose logging

        # --- Update and Display FPS ---
        fps_frame_count += 1
        # Update FPS approximately every second
        if time.time() - fps_start_time >= 1.0: 
            current_fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0 # Reset frame count
            fps_start_time = time.time() # Reset start time

        # --- Display Info Overlay on Frame ---
        info_lines = [
            f"FPS: {current_fps:.1f}",
            f"Objects: {detections_in_frame}",
            f"Conf: {current_conf_thresh_float:.2f} | IoU: {current_iou_thresh_float:.2f}",
            f"Status: {'PAUSED' if detection_paused else 'RUNNING'}",
            f"Save Log: {'ON' if save_detections_to_file else 'OFF'}",
            f"Filter: {', '.join([names[c] for c in filtered_classes]) if filtered_classes else 'ALL'}"
        ]
        
        # Draw each line of info on the frame
        for i, line in enumerate(info_lines):
            # Text position (x, y) - adjust as needed
            text_pos = (10, 30 + i * 25) 
            cv2.putText(frame, line, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # If detection is paused, overlay "PAUSED" in the center
        if detection_paused:
            (w_text, h_text), _ = cv2.getTextSize("PAUSED", cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
            cv2.putText(frame, "PAUSED", ((img_w - w_text) // 2, (img_h + h_text) // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

        # Show the processed frame in the OpenCV window
        cv2.imshow("YOLOv5 Detection", frame)
    
    else:
        # If webcam is not active, display a waiting message and controls
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Display main prompt
        main_text = "Press 's' to start webcam..."
        (text_w, text_h), _ = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = (blank_frame.shape[1] - text_w) // 2
        text_y = (blank_frame.shape[0] + text_h) // 2
        cv2.putText(blank_frame, main_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display controls list
        control_lines = [
            "--- Controls ---",
            " 's' : Start Webcam",
            " 'q' / 'ESC' : Quit Application",
            " 'c' : Capture Snapshot",
            " 'p' : Toggle Pause/Resume Detection",
            " 'f' : Toggle Save Detections to TXT file",
            " '0'-'9' : Toggle detection for specific classes",
            "-----------------"
        ]
        
        for i, line in enumerate(control_lines):
            cv2.putText(blank_frame, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("YOLOv5 Detection", blank_frame)