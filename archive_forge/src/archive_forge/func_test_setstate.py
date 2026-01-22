import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_setstate(self):
    """Verify that datetime dtype __setstate__ can handle bad arguments"""
    dt = np.dtype('>M8[us]')
    assert_raises(ValueError, dt.__setstate__, (4, '>', None, None, None, -1, -1, 0, 1))
    assert_(dt.__reduce__()[2] == np.dtype('>M8[us]').__reduce__()[2])
    assert_raises(TypeError, dt.__setstate__, (4, '>', None, None, None, -1, -1, 0, ({}, 'xxx')))
    assert_(dt.__reduce__()[2] == np.dtype('>M8[us]').__reduce__()[2])