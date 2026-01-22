import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_more_mixed_type(self):
    g = r_[-10.1, np.array([1]), np.array([2, 3, 4]), 10.0]
    assert_(g.dtype == 'f8')