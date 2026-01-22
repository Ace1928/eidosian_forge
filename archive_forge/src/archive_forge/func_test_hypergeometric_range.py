import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
def test_hypergeometric_range(self):
    assert_(np.all(random.hypergeometric(3, 18, 11, size=10) < 4))
    assert_(np.all(random.hypergeometric(18, 3, 11, size=10) > 0))
    args = [(2 ** 20 - 2, 2 ** 20 - 2, 2 ** 20 - 2)]
    is_64bits = sys.maxsize > 2 ** 32
    if is_64bits and sys.platform != 'win32':
        args.append((2 ** 40 - 2, 2 ** 40 - 2, 2 ** 40 - 2))
    for arg in args:
        assert_(random.hypergeometric(*arg) > 0)