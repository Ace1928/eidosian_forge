import numpy as np
from numpy.testing import assert_almost_equal
from pytest import raises as assert_raises
import scipy.optimize
def test_lmdif_errmsg(self):

    class SomeError(Exception):
        pass
    counter = [0]

    def func(x):
        counter[0] += 1
        if counter[0] < 3:
            return x ** 2 - np.array([9, 10, 11])
        else:
            raise SomeError()
    assert_raises(SomeError, scipy.optimize.leastsq, func, [1, 2, 3])