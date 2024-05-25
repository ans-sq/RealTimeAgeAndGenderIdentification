import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)

# Define paths to pre-trained models
faceProto = 'opencv_face_detector.pbtxt'
faceModel = 'opencv_face_detector_uint8.pb'
ageProto = 'age_deploy.prototxt'
ageModel = 'age_net.caffemodel'
genderProto = 'gender_deploy.prototxt'
genderModel = 'gender_net.caffemodel'

# Load pre-trained models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Define age and gender labels
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']
model_mean_value = (78.4263377603, 87.7689143744, 114.895847746)


# Define face detection and age/gender prediction functions
def faceBox(faceNet, frame):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frame_width)
            y1 = int(detection[0, 0, i, 4] * frame_height)
            x2 = int(detection[0, 0, i, 5] * frame_width)
            y2 = int(detection[0, 0, i, 6] * frame_height)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs


def detect_age_gender():
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        frame, bboxs = faceBox(faceNet, frame)
        for bbox in bboxs:
            face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), model_mean_value, swapRB=False)
            genderNet.setInput(blob)
            gender_pred = genderNet.forward()
            gender = gender_list[gender_pred[0].argmax()]
            ageNet.setInput(blob)
            age_pred = ageNet.forward()
            age = age_list[age_pred[0].argmax()]
            label = "{},{}".format(gender, age)
            cv2.rectangle(frame, (bbox[0], bbox[1] - 10), (bbox[2], bbox[1]), (0, 255, 0), -1)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


# Define Flask routes
@app.route('/')
def index():
    return render_template('index.html', video=False)


@app.route('/video')
def real_time_feed():
    return render_template('index.html', video=True)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/developers')
def developers():
    return render_template('developer.html')


@app.route('/video_feed')
def video_feed():
    return Response(detect_age_gender(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
