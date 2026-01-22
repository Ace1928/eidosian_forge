import numpy as np
import tensorflow as tf
from autokeras.engine import analyser
def correct_shape(self):
    if len(self.shape) == 1:
        return True
    return len(self.shape) == 2 and self.shape[1] == 1