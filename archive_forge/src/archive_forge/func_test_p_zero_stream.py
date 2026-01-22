import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
def test_p_zero_stream(self):
    np.random.seed(12345)
    assert_array_equal(random.binomial(1, [0, 0.25, 0.5, 0.75, 1]), [0, 0, 0, 1, 1])