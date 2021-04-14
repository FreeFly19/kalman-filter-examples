import cv2

from .base_filter import BaseFilter

import numpy as np


class OpencvKalmanFilter(BaseFilter):

    def __init__(self):
        super().__init__()

        # KalmanFilter(stateVectorSize, measurementVectorSize)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)

        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)

        self.kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32) * 0.5

        self.kf.measurementNoiseCov = np.array([[1, 0],
                                                [0, 1]], np.float32) * 0.5

    def update(self, position):
        self.kf.correct(position.astype(np.float32))
        return self.kf.statePost

    def predict(self):
        return self.kf.predict()

    def uncertainty(self):
        return (self.kf.errorCovPost[0][0], self.kf.errorCovPost[1][1])

    def color(self):
        return (0, 255, 255)

    def size(self):
        return 6
