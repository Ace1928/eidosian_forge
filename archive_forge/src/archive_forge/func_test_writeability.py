import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_writeability(self):
    x, y = np.unravel_index([1, 2, 3], (4, 5))
    assert_(x.flags.writeable)
    assert_(y.flags.writeable)