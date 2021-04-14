from .base_filter import BaseFilter

import numpy as np


class DumbFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        self._last_position = np.array([0, 0])

    def update(self, position):
        self._last_position = position
        return position

    def predict(self):
        return self._last_position

    def uncertainty(self):
        return (0, 0)

    def color(self):
        return (255, 120, 0)

    def size(self):
        return 10
