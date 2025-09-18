from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
import os
from datetime import datetime
import face_recognition
import dlib
from object_detection import yoloV3Detect
from landmark_models import *
from face_spoofing import *
from headpose_estimation import *
from face_detection import get_face_detector, find_faces
from collections import Counter

app = Flask(__name__)
CORS(app)

# Initialize models
h_model = load_hp_model('models/Headpose_customARC_ZoomShiftNoise.hdf5')
face_model = get_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Load known faces
known_face_encodings = []
known_face_names = []
for image in os.listdir('student_db'):
    face_image = face_recognition.load_image_file('student_db/' + image)
    face_encoding = face_recognition.face_encodings(face_image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(image.split('.')[0])

# Store proctoring logs
proctoring_logs = []

def process_frame(frame):
    frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image. Check if the image data is valid.")
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    log = {
        "timestamp": datetime.now().isoformat(),
        "num_people": 0,
        "banned_objects": [],
        "face_recognition": "No face detected",
        "head_pose": {"pitch": 0, "yaw": 0, "roll": 0, "status": "Normal"},
        "eye_tracking": "Looking at screen",
        "mouth_open": False,
        "face_spoofing": False,
        "alerts": []
    }

    try:
        # Object Detection
        fboxes, fclasses = yoloV3Detect(small_frame)
        to_detect = ['person', 'laptop', 'cell phone', 'book', 'tv']
        filtered_classes = [cls for cls in fclasses if cls in to_detect]
        count_items = Counter(filtered_classes)
        log["num_people"] = count_items.get("person", 0)
        log["banned_objects"] = [cls for cls in ['laptop', 'cell phone', 'book', 'tv'] 
                               if count_items.get(cls, 0) > 0]

        if log["num_people"] != 1:
            log["alerts"].append("Multiple people detected")
        if log["banned_objects"]:
            log["alerts"].append(f"Banned objects detected: {', '.join(log['banned_objects'])}")

        # Face Detection
        faces = find_faces(small_frame, face_model)
        if faces:
            face = faces[0]
            face_locations = [[face[1], face[2], face[3], face[0]]]
            
            # Face Recognition
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            if face_encodings:
                face_encoding = face_encodings[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                log["face_recognition"] = known_face_names[best_match_index] if matches[best_match_index] else "Unknown"
                
                if log["face_recognition"] == "Unknown":
                    log["alerts"].append("Unknown face detected")

            # Prepare for landmark detection
            left, top, right, bottom = face[0]*4, face[1]*4, face[2]*4, face[3]*4
            face_dlib = dlib.rectangle(left, top, right, bottom)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = predictor(gray, face_dlib)

            # Mouth Detection
            mouth_ratio = get_mouth_ratio([60, 62, 64, 66], frame, landmarks)
            log["mouth_open"] = mouth_ratio > 0.1
            if log["mouth_open"]:
                log["alerts"].append("Mouth movement detected")

            # Head Pose Estimation
            oAnglesNp, _ = headpose_inference(h_model, frame, face)
            log["head_pose"] = {
                "pitch": float(oAnglesNp[0]),
                "yaw": float(oAnglesNp[1]),
                "roll": float(oAnglesNp[2]),
                "status": "Normal"
            }
            
            # More accurate head pose detection
            if (abs(oAnglesNp[0]) > 15 or abs(oAnglesNp[1]) > 15):
                log["head_pose"]["status"] = "Suspicious"
                log["alerts"].append("Suspicious head movement detected")

            # Eye Tracking
            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], frame, landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], frame, landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
            
            gaze_ratio1_left_eye, _ = get_gaze_ratio([36, 37, 38, 39, 40, 41], frame, landmarks)
            gaze_ratio1_right_eye, _ = get_gaze_ratio([42, 43, 44, 45, 46, 47], frame, landmarks)
            gaze_ratio1 = (gaze_ratio1_right_eye + gaze_ratio1_left_eye) / 2
            
            if gaze_ratio1 <= 0.35:
                log["eye_tracking"] = "Looking right"
                log["alerts"].append("Looking away from screen (right)")
            elif gaze_ratio1 >= 4:
                log["eye_tracking"] = "Looking left"
                log["alerts"].append("Looking away from screen (left)")
            else:
                log["eye_tracking"] = "Looking at screen"

            # Face Spoofing Detection
            measures = face_spoof(frame, face)
            log["face_spoofing"] = np.mean(measures) < 0.7
            if log["face_spoofing"]:
                log["alerts"].append("Possible spoof face detected")

        else:
            log["alerts"].append("No face detected")

    except Exception as e:
        log["alerts"].append(f"Processing error: {str(e)}")
        # Ensure all keys are present even on error
        if "head_pose" not in log:
            log["head_pose"] = {"pitch": 0, "yaw": 0, "roll": 0, "status": "Error"}
        if "eye_tracking" not in log:
            log["eye_tracking"] = "Unknown"
        if "mouth_open" not in log:
            log["mouth_open"] = False
        if "face_spoofing" not in log:
            log["face_spoofing"] = False
        if "face_recognition" not in log:
            log["face_recognition"] = "No face detected"
        if "banned_objects" not in log:
            log["banned_objects"] = []
        if "num_people" not in log:
            log["num_people"] = 0

    proctoring_logs.append(log)
    return to_python_type(log)

def to_python_type(obj):
    """Recursively convert numpy types in dicts/lists to native Python types."""
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

@app.route('/api/capture', methods=['POST'])
def capture_face():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        result = process_frame(base64.b64decode(data['image']))
        # Ensure all numpy types are converted to native Python types
        result = to_python_type(result)
        return jsonify(result)
    except Exception as e:
        import traceback
        print("Error in /api/capture:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs', methods=['GET'])
def get_logs():
    triggered_logs = [log for log in proctoring_logs if log["alerts"]]
    return jsonify({
        "logs": triggered_logs,
        "total_logs": len(triggered_logs)
    })

@app.route('/api/clear_logs', methods=['POST'])
def clear_logs():
    global proctoring_logs
    proctoring_logs = []
    return jsonify({"status": "Logs cleared"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)