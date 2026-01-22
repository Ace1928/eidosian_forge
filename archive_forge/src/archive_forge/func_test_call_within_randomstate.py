import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
def test_call_within_randomstate(self):
    m = random.RandomState()
    res = np.array([0, 8, 7, 2, 1, 9, 4, 7, 0, 3])
    for i in range(3):
        random.seed(i)
        m.seed(4321)
        assert_array_equal(m.choice(10, size=10, p=np.ones(10) / 10.0), res)