from .base_filter import BaseFilter

import numpy as np


class KalmanFilter(BaseFilter):
    def __init__(self, process_noise_std, measurement_noise_std):
        super().__init__()

        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std

        self.x = np.array([0, 0, 0, 0])

        self.P = np.eye(4) * 5

        self.A = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        # Control vector, not needed for now
        self.u = np.array([0, 0])

        # Control to state increment
        self.B = np.array([[1 / 2, 0],
                           [0, 1 / 2],
                           [1, 0],
                           [0, 1]])

        # Measurement Mapping Matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Initial Process Noise Covariance
        self.Q = np.array([[1 / 4, 0, 1 / 2, 0],
                           [0, 1 / 4, 0, 1 / 2],
                           [1 / 2, 0, 1, 0],
                           [0, 1 / 2, 0, 1]]) * self.process_noise_std ** 2

        # Initial Measurement Noise Covariance
        self.R = np.array([[self.measurement_noise_std ** 2, 0],
                           [0, self.measurement_noise_std ** 2]])

    def update(self, position):
        S = self.H @ self.P @ self.H.T + self.R

        K = self.P @ self.H.T @ np.linalg.pinv(S)

        self.x = self.x + K @ (position - self.H @ self.x)
        I = np.eye(self.H.shape[1])
        # Update error covariance matrix
        self.P = (I - (K @ self.H)) @ self.P
        return self.x[0:2]


    def predict(self):
        self.x = self.A @ self.x + self.B @ self.u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[0:2]

    def uncertainty(self):
        return (int(self.P[0][0]**0.5), int(self.P[1][1]**0.5))

    def color(self):
        return (0, 255, 0)

    def size(self):
        return 6
