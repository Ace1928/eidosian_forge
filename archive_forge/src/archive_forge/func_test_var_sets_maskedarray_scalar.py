import numpy as np
from numpy.testing import (
def test_var_sets_maskedarray_scalar(self):
    a = np.ma.array(np.arange(5), mask=True)
    mout = np.ma.array(-1, dtype=float)
    a.var(out=mout)
    assert_(mout._data == 0)