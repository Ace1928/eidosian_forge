import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_einsum_failed_on_p9_and_s390x(self):
    tensor = np.random.random_sample((10, 10, 10, 10))
    x = np.einsum('ijij->', tensor)
    y = tensor.trace(axis1=0, axis2=2).trace()
    assert_allclose(x, y)