from flask import Flask, render_template, Response
import cv2
import random

app = Flask(__name__)

camera = cv2.VideoCapture(0)
camera.set(3, 640)  # width
camera.set(4, 480)  # height

emotions = ["Happy", "Sad", "Angry"]

music = {
    "Happy": "happy.mp3",
    "Sad": "sad.mp3",
    "Angry": "angry.mp3"
}

def generate_frames():
    while True:
        success, frame = camera.read()

        if not success:
            print("Failed to grab frame")
            break

        # Flip camera
        frame = cv2.flip(frame, 1)

        # Increase brightness
        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)

        # Add text
        cv2.putText(frame, "Camera OK", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    emotion = random.choice(emotions)
    song = music[emotion]
    return render_template('index.html', emotion=emotion, song=song)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
