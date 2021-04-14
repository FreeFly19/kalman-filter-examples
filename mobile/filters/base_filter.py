import abc

import numpy as np


class BaseFilter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, position):
        raise Exception()

    @abc.abstractmethod
    def predict(self):
        raise Exception()

    @abc.abstractmethod
    def uncertainty(self):
        raise Exception()

    @abc.abstractmethod
    def color(self):
        raise Exception()

    @abc.abstractmethod
    def size(self):
        raise Exception()
