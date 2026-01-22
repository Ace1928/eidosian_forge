import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_empty_array_unravel(self):
    res = np.unravel_index(np.zeros(0, dtype=np.intp), (2, 1, 0))
    assert len(res) == 3
    assert all((a.shape == (0,) for a in res))
    with assert_raises(ValueError):
        np.unravel_index([1], (2, 1, 0))