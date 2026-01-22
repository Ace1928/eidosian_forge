import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
def test_do_not_rewrite_previous_keyword(self):
    with np.errstate(divide='ignore', invalid='ignore'):
        vals = nan_to_num(np.array((-1.0, 0, 1)) / 0.0, nan=np.inf, posinf=999)
    assert_all(np.isfinite(vals[[0, 2]]))
    assert_all(vals[0] < -10000000000.0)
    assert_equal(vals[[1, 2]], [np.inf, 999])
    assert_equal(type(vals), np.ndarray)