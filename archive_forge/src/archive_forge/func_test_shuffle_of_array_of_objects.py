import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
def test_shuffle_of_array_of_objects(self):
    random.seed(1234)
    a = np.array([np.arange(1), np.arange(4)], dtype=object)
    for _ in range(1000):
        random.shuffle(a)
    import gc
    gc.collect()