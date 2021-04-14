from .base_filter import BaseFilter

import numpy as np


class LinearDisplacementFilter(BaseFilter):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.state = np.array([0, 0])
        self.velocity = np.array([0, 0])
        self.acceleration = np.array([0, 0])
        self.prev_state_steps = 1

    def update(self, position):
        prediction = self.predict()

        old_velocity = self.velocity

        self.velocity = (position - prediction) / self.prev_state_steps
        self.acceleration = (self.velocity - old_velocity) / self.prev_state_steps

        self.state = prediction * self.alpha + (1 - self.alpha) * position
        self.prev_state_steps = 1

        return self.state

    def predict(self):
        self.state = self.state + self.velocity + 0.5 * self.acceleration**2
        self.prev_state_steps += 1
        return self.state

    def uncertainty(self):
        return (0, 0)

    def color(self):
        return (150, 255, 0)

    def size(self):
        return 8
