import numpy as np
from numpy.testing import (
def test_arraytypes_fasttake(self):
    """take from a 0-length dimension"""
    x = np.empty((2, 3, 0, 4))
    assert_raises(IndexError, x.take, [0], axis=2)
    assert_raises(IndexError, x.take, [1], axis=2)
    assert_raises(IndexError, x.take, [0], axis=2, mode='wrap')
    assert_raises(IndexError, x.take, [0], axis=2, mode='clip')