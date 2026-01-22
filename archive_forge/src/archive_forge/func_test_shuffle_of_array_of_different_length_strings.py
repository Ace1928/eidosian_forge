import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
def test_shuffle_of_array_of_different_length_strings(self):
    random.seed(1234)
    a = np.array(['a', 'a' * 1000])
    for _ in range(100):
        random.shuffle(a)
    import gc
    gc.collect()