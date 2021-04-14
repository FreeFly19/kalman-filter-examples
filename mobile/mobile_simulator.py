import time

import numpy as np

import cv2

from mobile.filters.base_filter import BaseFilter


class MobileAPI:
    def __init__(self, pos_measurement_variance, acceleration_change_frequency, measurement_frequency):
        self._position = np.array([0, 0])
        self._velocity = np.array([0, 0])
        self._acceleration = np.array([0, 0])
        self._last_measured_pos = self._position

        self.pos_measurement_variance = pos_measurement_variance
        self.acceleration_change_frequency = acceleration_change_frequency
        self.measurement_frequency = measurement_frequency

        self.updates_count = 0

    @property
    def last_measured_position(self):
        return self._last_measured_pos

    @property
    def position(self):
        return self._position

    def get_accelerometer_data(self):
        return self._acceleration.astype(np.float32)

    # Private method, cannot be access from outside
    def update(self):
        if self.updates_count % self.acceleration_change_frequency == 0:
            self._acceleration = self._acceleration * 0.05 + np.random.normal(0, 1, 2)

        self._velocity = self._velocity * 0.8 + self._acceleration
        self._position = self.position + self._velocity + np.sign(self._acceleration) * (self._acceleration ** 2) / 2

        if self.updates_count % self.measurement_frequency == 0:
            self._last_measured_pos = self._position + np.random.normal(0, self.pos_measurement_variance, 2)

        self.updates_count += 1


class Simulator:
    def __init__(self, mobile: MobileAPI, filters: [BaseFilter]):
        self._mobile = mobile
        self._filters = filters
        self.show_real_position = True

        # renderer
        self.frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        self.camera = np.array([
            [1, 0, self.frame.shape[0] / 2],
            [0, 1, self.frame.shape[1] / 2],
            [0, 0, 0]
        ], dtype=float)

    def step(self):
        self._mobile.update()

        if -self.camera[0][2] >= self._mobile.position[0] or self._mobile.position[0] >= self.camera[0][2] or \
                -self.camera[1][2] >= self._mobile.position[1] or self._mobile.position[1] >= self.camera[1][2]:
            self._mobile = MobileAPI(self._mobile.pos_measurement_variance,
                                     self._mobile.acceleration_change_frequency,
                                     self._mobile.measurement_frequency)

        for f in self._filters:
            if (self._mobile.updates_count - 1) % self._mobile.measurement_frequency == 0:
                f.update(self._mobile.last_measured_position)

        self.render()

    def render(self):
        self.frame[:, :, :] = 0

        cv2.circle(self.frame, self.project_to_camera(self._mobile.last_measured_position), 6, (100, 100, 100), -1)

        if self.show_real_position:
            cv2.circle(self.frame, self.project_to_camera(self._mobile.position), 5, (0, 0, 255), -1)

        t = time.time()
        for f in self._filters:
            pos = f.predict()
            projected_point = self.project_to_camera(pos)

            alpha = 0.6
            alpha2 = 0.3
            overlay = self.frame.copy()
            overlay2 = self.frame.copy()
            cv2.circle(overlay, projected_point, f.size(), f.color(), -1)
            cv2.ellipse(overlay2, projected_point, f.uncertainty(), 0, 0, 360, f.color(), -1)
            self.frame = cv2.addWeighted(overlay, alpha, self.frame, 1 - alpha, 0)
            self.frame = cv2.addWeighted(overlay2, alpha2, self.frame, 1 - alpha2, 0)
        print(time.time() - t)

        cv2.imshow("Simulation", self.frame)
        cv2.waitKey(999999)

    def project_to_camera(self, point):
        projected_point = (self.camera @ np.array([point[0], point[1], 1])).astype(int)
        return (projected_point[1], projected_point[0])
