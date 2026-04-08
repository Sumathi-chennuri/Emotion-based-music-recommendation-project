from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
app = Flask(__name__)
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
)
music = {
    "Happy": "happy.mp3",
    "Sad": "sad.mp3",
    "Angry": "angry.mp3"
}
current_emotion = "Detecting..."
def detect_emotion(face_region):
    face_region = cv2.equalizeHist(face_region)
    smiles = smile_cascade.detectMultiScale(
        face_region,
        scaleFactor=1.3,
        minNeighbors=5
    )
    brightness = np.mean(face_region)
    if len(smiles) > 0:
        return "Happy"
    elif brightness > 100:
        return "Sad"
    else:
        return "Angry"
def generate_frames():
    global current_emotion
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detected = False
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face_detected = True
            face_region = gray[y:y+h, x:x+w]
            emotion = detect_emotion(face_region)
            current_emotion = emotion
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if not face_detected:
            current_emotion = "Detecting..."
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/get_emotion')
def get_emotion():
    song = music.get(current_emotion, "happy.mp3")
    return jsonify({
        "emotion": current_emotion,
        "song": song
    })
@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    app.run(debug=True)