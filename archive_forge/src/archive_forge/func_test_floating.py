import pytest
import numpy as np
from numpy.testing import (
def test_floating(self):
    fsingle = np.single('1.234')
    fdouble = np.double('1.234')
    flongdouble = np.longdouble('1.234')
    assert_almost_equal(fsingle, 1.234)
    assert_almost_equal(fdouble, 1.234)
    assert_almost_equal(flongdouble, 1.234)