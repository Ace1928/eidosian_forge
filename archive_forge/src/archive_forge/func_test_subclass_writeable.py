import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_subclass_writeable(self):
    d = np.rec.array([('NGC1001', 11), ('NGC1002', 1.0), ('NGC1003', 1.0)], dtype=[('target', 'S20'), ('V_mag', '>f4')])
    ind = np.array([False, True, True], dtype=bool)
    assert_(d[ind].flags.writeable)
    ind = np.array([0, 1])
    assert_(d[ind].flags.writeable)
    assert_(d[...].flags.writeable)
    assert_(d[0].flags.writeable)