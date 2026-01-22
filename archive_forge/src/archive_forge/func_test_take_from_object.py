import numpy as np
from numpy.testing import (
def test_take_from_object(self):
    d = np.zeros(5, dtype=object)
    assert_raises(IndexError, d.take, [6])
    d = np.zeros((5, 0), dtype=object)
    assert_raises(IndexError, d.take, [1], axis=1)
    assert_raises(IndexError, d.take, [0], axis=1)
    assert_raises(IndexError, d.take, [0])
    assert_raises(IndexError, d.take, [0], mode='wrap')
    assert_raises(IndexError, d.take, [0], mode='clip')