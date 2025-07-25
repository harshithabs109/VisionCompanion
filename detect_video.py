import torch
import cv2
import os
import uuid

model = torch.hub.load('./yolov5', 'yolov5s', source='local')

def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    result_filename = f"result_{uuid.uuid4().hex}_{os.path.basename(video_path)}"
    result_path = os.path.join("static", "results", result_filename)
    out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        out.write(results.render()[0])
    cap.release()
    out.release()
    return result_path
