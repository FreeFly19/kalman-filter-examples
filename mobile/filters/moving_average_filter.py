from .base_filter import BaseFilter

import numpy as np


class MovingAverageFilter(BaseFilter):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.positions = np.array([[0, 0]])

    def update(self, position):
        self.positions = np.concatenate((self.positions, np.array([position])), axis=0)[-self.window_size:]
        return position

    def predict(self):
        return self.positions.mean(axis=0)

    def uncertainty(self):
        return (0, 0)

    def color(self):
        return (255, 0, 255)

    def size(self):
        return 10
