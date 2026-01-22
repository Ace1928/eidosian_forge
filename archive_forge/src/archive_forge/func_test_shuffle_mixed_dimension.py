import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
def test_shuffle_mixed_dimension(self):
    for t in [[1, 2, 3, None], [(1, 1), (2, 2), (3, 3), None], [1, (2, 2), (3, 3), None], [(1, 1), 2, 3, None]]:
        random.seed(12345)
        shuffled = list(t)
        random.shuffle(shuffled)
        expected = np.array([t[0], t[3], t[1], t[2]], dtype=object)
        assert_array_equal(np.array(shuffled, dtype=object), expected)