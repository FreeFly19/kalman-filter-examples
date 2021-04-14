import cv2
import numpy as np

kf = cv2.KalmanFilter(8, 4)

kf.transitionMatrix = np.array([
    [1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
], dtype=np.float32)

kf.measurementMatrix = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
], dtype=np.float32)

kf.processNoiseCov = np.eye(8).astype(np.float32) * 0.5
kf.measurementNoiseCov = np.eye(4).astype(np.float32) * 50


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) # Alien?

# Link to download the video https://ssyoutube.com/watch?v=gFy6LVlNc6c
video_capture = cv2.VideoCapture('alien.mp4')
video_capture.set(cv2.CAP_PROP_POS_FRAMES, 130)

while True:
    # video_capture.read()
    _, frame = video_capture.read()
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    box = None
    max_w = 0.95
    for b, w in zip(boxes, weights):
        if max_w < w:
            max_w = w
            box = b

    if box is not None:
        kf.correct(np.array([box[0], box[1], box[2], box[3]], dtype=np.float32))
        left, top, right, bottom = box[0], box[1], box[0] + box[2], box[1] + box[3]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 4)

    state = kf.predict()
    if state[0] != 0:
        if kf.errorCovPost.sum() < 100000:
            left, top, right, bottom = state[0], state[1], state[0] + state[4], state[1] + state[5]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), int(20 - (kf.errorCovPost.sum() / 100000) * 20))

    cv2.imshow("Tracker", frame)
    cv2.waitKey(10)
