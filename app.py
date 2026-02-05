# install dependancies
from flask import Flask , render_template, Response, jsonify, request, redirect, url_for
import cv2
import time
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from collections import deque
import os


# ==============================
# APP SETUP
# ==============================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# DEVICE & MODEL
# ==============================
device = torch.device("cpu")

from torchvision.models import resnet18, ResNet18_Weights
resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()
resnet.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_features(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        features = resnet(input_tensor)

    return features.squeeze().cpu().numpy()

# ==============================
# BEHAVIOR CLASSIFICATION
# ==============================
def classify_behavior(feature_buffer):
    if len(feature_buffer) < 16:
        return "Monitoring"

    features = np.array(feature_buffer)
    motion = np.mean(np.abs(np.diff(features, axis=0)))

    if motion > 0.25:
        return "Abnormal Speed"
    elif motion > 0.12:
        return "Sudden Dispersion"
    else:
        return "Monitoring"

# ==============================
# MULTI-CAMERA SETUP
# ==============================
CAMERA_SOURCES = {
    
    1: r"F:\naveen\Project_Own\Multi-Camera Crowd Behavior\Videos\5286466-hd_1920_1080_30fps.mp4",
    2: r"F:\naveen\Project_Own\Multi-Camera Crowd Behavior\Videos\5286504-hd_1920_1080_30fps.mp4"
}

caps = {}
FEATURE_BUFFERS = {}
LAST_BEHAVIOR = {}
INFERENCE_INTERVAL = 5

for cid, src in CAMERA_SOURCES.items():
    if isinstance(src, int):
        # Skip webcam on cloud
        continue

    caps[cid] = cv2.VideoCapture(src)
    FEATURE_BUFFERS[cid] = deque(maxlen=16)
    LAST_BEHAVIOR[cid] = "Monitoring"

# ==============================
# LIVE CAMERA STREAM
# ==============================
def generate_frames(cam_id):
    cap = caps[cam_id]
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        frame = cv2.resize(frame, (640, 360))

        if frame_count % INFERENCE_INTERVAL == 0:
            features = extract_features(frame)
            FEATURE_BUFFERS[cam_id].append(features)
            behavior = classify_behavior(FEATURE_BUFFERS[cam_id])
            LAST_BEHAVIOR[cam_id] = behavior
        else:
            behavior = LAST_BEHAVIOR[cam_id]

        cv2.putText(
            frame,
            f"Camera {cam_id} | {behavior}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if behavior == "Monitoring" else (0, 0, 255),
            2
        )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

# ==============================
# UPLOADED VIDEO STREAM (ANALYSIS)
# ==============================
def generate_processed_frames(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    cap = cv2.VideoCapture(video_path)
    feature_buffer = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 360))
        features = extract_features(frame)
        feature_buffer.append(features)

        label = classify_behavior(feature_buffer)
        color = (0, 255, 0) if label == "Monitoring" else (0, 0, 255)

        cv2.putText(
            frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3
        )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()

# ==============================
# ROUTES
# ==============================
@app.route("/")
def index():
    return render_template("index.html", cameras=CAMERA_SOURCES.keys())

@app.route("/video_feed/<int:cam_id>")
def video_feed(cam_id):
    return Response(
        generate_frames(cam_id),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/analyze", methods=["POST"])
def analyze_video():
    file = request.files["video"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    return redirect(url_for("analysis_result", filename=file.filename))

@app.route("/analysis/<filename>")
def analysis_result(filename):
    return render_template("analysis.html", filename=filename)

@app.route("/processed_video/<filename>")
def processed_video(filename):
    return Response(
        generate_processed_frames(filename),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
