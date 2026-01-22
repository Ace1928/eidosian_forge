import os
import numpy as np
from numpy.testing import (
def test_mem_vectorise(self):
    vt = np.vectorize(lambda *args: args)
    vt(np.zeros((1, 2, 1)), np.zeros((2, 1, 1)), np.zeros((1, 1, 2)))
    vt(np.zeros((1, 2, 1)), np.zeros((2, 1, 1)), np.zeros((1, 1, 2)), np.zeros((2, 2)))