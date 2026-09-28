from flask import Flask, render_template, Response, jsonify
import cv2
import supervision as sv
from ultralytics import YOLO
import os
from drivemode import recommend_mode  # ✅ Use ML-based drive mode logic

# ✅ Explicitly define correct folders
app = Flask(__name__, template_folder="templates", static_folder="static")

# 🔧 Configuration
PR_MODEL_PATH = r"C:\Users\ynsai\Downloads\best (2).pt"
PR_VIDEO_PATH = r"C:\Users\ynsai\Downloads\video_preview_h264.mp4"

# 🌍 Global Variables
detection_count = 0
drive_mode = "Initializing..."
avg_confidence = 0.0
ml_mode_data = {}


class PyResearchVisualizer:
    def __init__(self):
        # ✅ Load YOLO model
        if not os.path.exists(PR_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {PR_MODEL_PATH}")
        print("🔍 Loading YOLO model...")
        self.model = YOLO(PR_MODEL_PATH)
        print("✅ Model loaded successfully")

        # 🖼️ Visual annotators
        self.box_annotator = sv.RoundBoxAnnotator(thickness=2, color=sv.Color.from_hex("#00FFFF"))
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.7,
            text_thickness=1,
            text_color=sv.Color.WHITE,
            text_padding=10
        )

    def process_frame(self, frame):
        global detection_count, drive_mode, avg_confidence, ml_mode_data

        # Resize for consistency
        frame = cv2.resize(frame, (640, 360))

        # YOLO detection
        results = self.model.predict(frame, verbose=False, conf=0.25, iou=0.45, max_det=500)[0]

        # Detection info
        detections = sv.Detections.from_ultralytics(results)
        detection_count = len(detections)

        # Average confidence
        if len(results.boxes.conf) > 0:
            avg_confidence = float(results.boxes.conf.mean() * 100)
        else:
            avg_confidence = 0.0

        # ✅ ML-based drive mode recommendation with severity
        ml_mode_data = recommend_mode(detection_count, avg_confidence, 30)
        drive_mode = ml_mode_data.get("mode", "Unknown")

        # Annotate frame
        annotated_frame = self.box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Overlay drive mode label with severity
        severity = ml_mode_data.get("severity", {})
        severity_text = severity.get("level", "Unknown")
        severity_color = severity.get("color_bgr", (0, 255, 255))
        
        cv2.rectangle(annotated_frame, (10, 10), (400, 60), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Drive Mode: {drive_mode}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Severity: {severity_text}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, severity_color, 1)

        return annotated_frame


def generate_frames():
    """Generator function to stream frames"""
    visualizer = PyResearchVisualizer()
    if not os.path.exists(PR_VIDEO_PATH):
        raise FileNotFoundError(f"Video file not found at: {PR_VIDEO_PATH}")

    cap = cv2.VideoCapture(PR_VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if fps > 0 else 0.03

    print(f"🎥 Streaming started at {fps:.2f} FPS...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("⚠️ End of video or cannot read frame.")
            break

        # Process YOLO frame
        output_frame = visualizer.process_frame(frame)

        # Convert for streaming
        try:
            _, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()
        except Exception as e:
            print("Frame encode error:", e)
            continue

        # Stream frame to browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cv2.waitKey(int(frame_delay * 1000))

    cap.release()
    print("🚫 Video stream ended.")


# 🖥️ Flask Routes
@app.route('/')
def index():
    """Render dashboard"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Stream live YOLO frames"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detection_count')
def get_detection_count():
    """Send live detection info with ML mode data and severity to frontend"""
    return jsonify({
        'detections': detection_count,
        'avg_confidence': round(avg_confidence, 2),
        'drive_mode': drive_mode,
        'ml_mode_data': ml_mode_data  # ✅ Send full ML data including severity
    })


# 🧪 Static image test route
@app.route('/test_images')
def test_images():
    """Verify static image accessibility"""
    return '''
    <h2>Static Image Test</h2>
    <img src="/static/img/amg.png" width="200"><br>
    <img src="/static/img/g_wagon.png" width="200"><br>
    <img src="/static/img/caution.png" width="200"><br>
    '''


if __name__ == "__main__":
    print("🌐 Starting Flask server at http://127.0.0.1:5000 ...")
    app.run(debug=True, host='0.0.0.0')