import cv2

from .base_filter import BaseFilter

import numpy as np


class OpencvKalmanFilterWithControl(BaseFilter):

    def __init__(self, accelerometer_data_getter):
        super().__init__()
        self.accelerometer_data_getter = accelerometer_data_getter

        #KalmanFilter(stateVectorSize, measurementVectorSize)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]], np.float32)

        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 0.8, 0],
                                            [0, 0, 0, 0.8]], np.float32)

        self.kf.measurementNoiseCov = np.array([[1, 0],
                                                [0, 1]], np.float32) * 5

        self.kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], np.float32) * 0.3

        self.kf.controlMatrix = np.array([[1/2, 0],
                                          [0, 1/2],
                                          [1, 0],
                                          [0, 1]
                                          ], np.float32)

    def update(self, position):
        self.kf.correct(position.astype(np.float32))

        return self.kf.statePost

    def predict(self):
        return self.kf.predict(self.accelerometer_data_getter())

    def uncertainty(self):
        return (self.kf.errorCovPost[0][0], self.kf.errorCovPost[1][1])

    def color(self):
        return (255, 0, 255)

    def size(self):
        return 6
