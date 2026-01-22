import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import pytest
from itertools import chain
def test_unpackbits_large():
    d = np.arange(277, dtype=np.uint8)
    assert_array_equal(np.packbits(np.unpackbits(d)), d)
    assert_array_equal(np.packbits(np.unpackbits(d[::2])), d[::2])
    d = np.tile(d, (3, 1))
    assert_array_equal(np.packbits(np.unpackbits(d, axis=1), axis=1), d)
    d = d.T.copy()
    assert_array_equal(np.packbits(np.unpackbits(d, axis=0), axis=0), d)