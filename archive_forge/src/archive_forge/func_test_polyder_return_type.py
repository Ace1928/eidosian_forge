import os
import numpy as np
from numpy.testing import (
def test_polyder_return_type(self):
    assert_(isinstance(np.polyder(np.poly1d([1]), 0), np.poly1d))
    assert_(isinstance(np.polyder([1], 0), np.ndarray))
    assert_(isinstance(np.polyder(np.poly1d([1]), 1), np.poly1d))
    assert_(isinstance(np.polyder([1], 1), np.ndarray))