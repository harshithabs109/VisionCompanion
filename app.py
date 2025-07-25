import cv2
import torch
import numpy as np
import os
from pathlib import Path
import sys
import uuid # For generating unique filenames
import logging # For better logging

# Flask imports
from flask import Flask, render_template, request, send_from_directory, redirect, url_for

# Configure logging for detailed output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Parameters (ADJUST THESE IF NECESSARY) ---
# Path to your cloned YOLOv5 repository.
# This assumes 'yolov5' folder is directly inside 'vision-companion'.
YOLOV5_REPO_PATH = Path("yolov5") 

# Path to the YOLOv5 weights file.
# This assumes 'yolov5s.pt' is directly inside your 'yolov5' folder.
WEIGHTS_PATH = YOLOV5_REPO_PATH / 'yolov5s.pt' 

# Folders for handling file uploads and results
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results' # Processed images/videos will be saved here and served

# Model input size (standard for YOLOv5)
IMG_SIZE = 640 
# Default confidence and IoU thresholds for detection
DEFAULT_CONF_THRESHOLD = 0.25 # Adjust if you need more/fewer detections
DEFAULT_IOU_THRESHOLD = 0.45 # Adjust for how much overlap is allowed

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'webm'} # Common video formats
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS.union(ALLOWED_VIDEO_EXTENSIONS)

# --- Setup YOLOv5 Path and Imports ---
# Add the YOLOv5 directory to the Python path to allow importing its modules
if not YOLOV5_REPO_PATH.is_dir():
    logger.error(f"Error: YOLOv5 directory not found at '{YOLOV5_REPO_PATH}'.")
    logger.error("Please ensure the 'yolov5' folder is in the same directory as this script.")
    logger.error("Or update the 'YOLOV5_REPO_PATH' variable to point to your YOLOv5 clone.")
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
    logger.error(f"\nError importing YOLOv5 modules: {e}")
    logger.error("Please ensure the 'yolov5' folder is correctly configured in sys.path.")
    logger.error(f"Also, run 'pip install -r {YOLOV5_REPO_PATH}/requirements.txt' to install all dependencies.")
    sys.exit(1)

# --- Ensure Required Folders Exist ---
# Create the upload and results directories if they don't already exist
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)
logger.info(f"Ensured '{UPLOAD_FOLDER}' and '{RESULTS_FOLDER}' directories exist.")

# --- Load YOLOv5 Model ---
if not WEIGHTS_PATH.is_file():
    logger.error(f"\nError: YOLOv5 weights file not found at '{WEIGHTS_PATH}'.")
    logger.error("Please ensure 'yolov5s.pt' exists inside your 'yolov5' folder.")
    logger.error("You might need to download it or move it from the parent directory.")
    sys.exit(1)

# Select device (CPU or GPU). '' for auto-select (prefers GPU if available).
device = select_device('') 

try:
    logger.info(f"Attempting to load YOLOv5 model from '{WEIGHTS_PATH}' on {device}...")
    model = attempt_load(WEIGHTS_PATH, device=device, inplace=True, fuse=True) 
    model.eval() # Set model to evaluation mode
    # Get class names from the loaded model
    names = model.module.names if hasattr(model, 'module') else model.names 
    logger.info(f"YOLOv5 Model loaded successfully. Found {len(names)} classes.")
except Exception as e:
    logger.error(f"\nError loading YOLOv5 model: {e}")
    logger.error("Possible reasons: Corrupted weights file, incompatible PyTorch version, or CUDA issues.")
    sys.exit(1)

# --- Flask App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
# Increased max content length for video files (e.g., 200MB)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024 

# --- Helper Function to Check File Type ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_image_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def is_video_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


# --- YOLOv5 Inference Function for Images ---
def detect_objects_on_image(image_path, conf_thres=DEFAULT_CONF_THRESHOLD, iou_thres=DEFAULT_IOU_THRESHOLD):
    """
    Performs object detection on a single image file.
    Returns the annotated image (NumPy array) and any error message.
    """
    logger.debug(f"Inside detect_objects_on_image. Image path received: {image_path}")
    
    if not Path(image_path).is_file():
        logger.error(f"Image file NOT FOUND at: {image_path}")
        return None, f"Error: Image file not found on the server at '{image_path}'."

    img = cv2.imread(str(image_path)) 
    
    if img is None:
        logger.error(f"cv2.imread returned None for path: {image_path}. File might be corrupted or an unsupported format.")
        return None, "Error: Could not read the image file. It might be corrupted or an unsupported format."
    
    # Convert BGR (OpenCV default) to RGB (YOLOv5 expects RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize image to model's expected input size
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)) 
    # Convert numpy array to PyTorch tensor, transpose to (C, H, W), normalize to 0-1
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    # Add batch dimension (B, C, H, W) and move to appropriate device (CPU/GPU)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad(): # Disable gradient calculation for inference
        pred = model(img_tensor)[0] # Get raw predictions from the model
        
        # Apply Non-Max Suppression (NMS)
        det = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)[0]
        
    logger.info(f"Image detection: Found {len(det)} objects with confidence >= {conf_thres:.2f} and IoU <= {iou_thres:.2f}.")
    
    annotator = Annotator(img, line_width=2, example=str(names))

    if len(det): # If detections are found
        # Rescale bounding box coordinates from IMG_SIZE to original image dimensions
        det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img.shape).round()

        # Draw bounding boxes and labels for each detected object
        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}' # Format label: "ClassName Confidence"
            annotator.box_label(xyxy, label, color=colors(int(cls), True))
    else: # If no detections, return the original image without annotations
        logger.info("No objects detected in image after NMS. Returning original image.")
        return img, None 

    # Get the annotated image (NumPy array)
    processed_image = annotator.result()
    return processed_image, None # Return annotated image and no error

# --- YOLOv5 Inference Function for Videos ---
def detect_objects_on_video(video_path, output_path, conf_thres=DEFAULT_CONF_THRESHOLD, iou_thres=DEFAULT_IOU_THRESHOLD):
    """
    Performs object detection on a video file frame by frame and saves the output video.
    Returns True on success, False and an error message on failure.
    """
    logger.info(f"Starting video detection for: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return False, "Error: Could not open the video file. It might be corrupted or an unsupported format."

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Check if video properties are valid
    if frame_width == 0 or frame_height == 0 or fps == 0:
        logger.error(f"Invalid video properties: width={frame_width}, height={frame_height}, fps={fps} for {video_path}")
        cap.release()
        return False, "Error: Could not read video properties (width, height, FPS). Video might be malformed."

    # Define the codec and create VideoWriter object
    # For MP4, 'mp4v' or 'avc1' are common codecs. 'mp4v' is generally safer for cross-platform.
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        logger.error(f"Could not create video writer for output: {output_path}. Check codec, permissions, or disk space.")
        cap.release()
        return False, "Error: Could not create output video file. Check permissions, disk space, or codec support."

    frame_count = 0
    detected_frames_count = 0 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream or error reading frame.")
            break # End of video or error reading frame

        frame_count += 1
        if frame_count % 100 == 0: # Log every 100 frames to avoid spamming terminal
            logger.debug(f"Processing frame {frame_count}...")

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)[0]
            det = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)[0]

        annotator = Annotator(frame, line_width=2, example=str(names))

        if len(det):
            detected_frames_count += 1 # Increment if detections are found
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(int(cls), True))
        
        annotated_frame = annotator.result()
        out.write(annotated_frame) # Write the annotated frame to the output video

    cap.release()
    out.release()
    logger.info(f"Video detection complete. Processed {frame_count} frames. Detected objects in {detected_frames_count} frames. Output saved to: {output_path}")
    
    if frame_count == 0: # If no frames were processed, it means video was unreadable
        return False, "Error: The video contained no readable frames or could not be processed."
    
    return True, None # Success

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Handles image/video uploads and triggers object detection.
    GET request: Displays the upload form (index.html).
    POST request: Processes the uploaded file.
    """
    if request.method == 'POST':
        # Check if a file was actually submitted
        if 'file' not in request.files:
            logger.warning("No 'file' part in the request.")
            return render_template('index.html', error="No file part in the request. Please select a file.")
        
        file = request.files['file']
        
        # Check if a file was selected by the user (filename is not empty)
        if file.filename == '':
            logger.warning("No file selected by user for upload.")
            return render_template('index.html', error="No file selected for upload. Please choose a file.")
        
        # Check for allowed file extensions
        if not allowed_file(file.filename):
            logger.warning(f"Uploaded file has unsupported extension: {file.filename}")
            return render_template('index.html', error=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}.")

        # If a file is present and allowed
        if file:
            original_filename = file.filename
            file_extension = Path(original_filename).suffix
            unique_filename = str(uuid.uuid4()) + file_extension
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            try:
                file.save(filepath) # Save the uploaded file temporarily
                logger.info(f"Uploaded file saved to: {filepath}")
            except Exception as e:
                logger.exception(f"ERROR: Failed to save uploaded file {filepath}.")
                return render_template('index.html', error=f"Failed to save uploaded file on server: {e}")

            # Verify the uploaded file actually exists after saving
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                logger.error(f"Uploaded file expected at {filepath} but not found or is empty after saving.")
                # Attempt to remove any residue, though likely none
                if os.path.exists(filepath): os.remove(filepath)
                return render_template('index.html', error="Uploaded file disappeared or is empty after saving. Please try again.")


            result_filename = "detected_" + unique_filename
            result_filepath = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            
            processing_successful = False
            detection_error = None
            is_video = False # Initialize as False

            if is_image_file(file.filename):
                logger.info(f"Processing image file: {original_filename}")
                processed_content, detection_error = detect_objects_on_image(filepath)
                is_video = False # Explicitly confirm for images
                
                if processed_content is not None:
                    if not isinstance(processed_content, np.ndarray) or processed_content.size == 0:
                        logger.error(f"Processed image content is invalid (not a NumPy array or empty) for {original_filename}.")
                        detection_error = "Detection produced invalid image content."
                    else:
                        try:
                            cv2.imwrite(result_filepath, processed_content)
                            if os.path.exists(result_filepath) and os.path.getsize(result_filepath) > 0:
                                processing_successful = True
                                logger.info(f"Processed image saved and verified: {result_filepath}")
                            else:
                                detection_error = "Processed image saved, but the output file is empty or corrupted."
                                logger.error(detection_error)
                        except Exception as e:
                            logger.exception(f"ERROR: Failed to save processed image {result_filepath}.")
                            detection_error = f"Failed to save processed image: {e}"
                else: 
                    logger.error(f"detect_objects_on_image returned None for content. Error: {detection_error}")


            elif is_video_file(file.filename):
                logger.info(f"Processing video file: {original_filename}")
                processing_successful, detection_error = detect_objects_on_video(filepath, result_filepath)
                is_video = True # Set to True for videos
                
                if processing_successful: 
                    if not os.path.exists(result_filepath) or os.path.getsize(result_filepath) == 0:
                        processing_successful = False
                        detection_error = "Video processing completed but output video file is missing or empty."
                        logger.error(detection_error)
                    else:
                        logger.info(f"Processed video saved and verified: {result_filepath}")
                else: 
                    logger.error(f"detect_objects_on_video indicated failure. Error: {detection_error}")

            else:
                detection_error = "Internal server error: File type not correctly identified for processing after initial checks."
                logger.error(detection_error)

            # Remove the temporarily uploaded file after processing
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # --- THIS BLOCK IS THE PROBLEM. IT'S IN THE WRONG PLACE AND INDENTATION. ---
            # if __name__ == '__main__':
            #     logger.info("Starting Flask application...")
            # # Run the Flask app in debug mode (turn off for production for security/performance)
            #     app.run(debug=True)
            # --- END OF PROBLEM BLOCK ---

            if not processing_successful:
                return render_template('index.html', error=f"Detection/Save failed: {detection_error}")

            # Render the result template, passing the URL of the processed file and its type
            return render_template('result.html', 
                                   file_url=f"/static/results/{result_filename}",
                                   is_video=is_video)
    
    # For GET requests (when the user first visits the page), render the upload form
    return render_template('index.html')

# --- THIS IS WHERE THE BLOCK SHOULD BE, WITH NO INDENTATION ---
if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=True)
# --- END CORRECT PLACEMENT ---