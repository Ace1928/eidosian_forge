import numpy as np
from numpy.testing import (
def test_ddof_corrcoef(self):
    x = np.ma.masked_equal([1, 2, 3, 4, 5], 4)
    y = np.array([2, 2.5, 3.1, 3, 5])
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning, 'bias and ddof have no effect')
        r0 = np.ma.corrcoef(x, y, ddof=0)
        r1 = np.ma.corrcoef(x, y, ddof=1)
        assert_allclose(r0.data, r1.data)