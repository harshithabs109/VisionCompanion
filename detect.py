# detect.py
import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path

# Add yolov5 path to sys
FILE = os.path.realpath(__file__)
ROOT = os.path.dirname(FILE)
YOLOV5_PATH = os.path.join(ROOT, 'yolov5')
sys.path.append(YOLOV5_PATH)

from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from models.common import DetectMultiBackend

def detect_objects(source='0', weights='yolov5s.pt', imgsz=640):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=True) if source == '0' else LoadImages(source, img_size=imgsz, stride=stride, auto=True)

    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        results = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in det:
                    label = f"{names[int(cls)]} {conf:.2f}"
                    results.append(label)
                    cv2.rectangle(im0s, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(im0s, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        yield im0s, results
