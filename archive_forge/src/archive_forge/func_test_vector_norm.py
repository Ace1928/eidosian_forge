import pytest
import numpy as np
from numpy.linalg import norm as npnorm
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import scipy.sparse
from scipy.sparse.linalg import norm as spnorm
def test_vector_norm(self):
    v = [4.58257569495584, 4.242640687119285, 4.58257569495584]
    for m, a in ((self.b, 0), (self.b.T, 1)):
        for axis in (a, (a,), a - 2, (a - 2,)):
            assert_allclose(spnorm(m, 1, axis=axis), [7, 6, 7])
            assert_allclose(spnorm(m, np.inf, axis=axis), [4, 3, 4])
            assert_allclose(spnorm(m, axis=axis), v)
            assert_allclose(spnorm(m, ord=2, axis=axis), v)
            assert_allclose(spnorm(m, ord=None, axis=axis), v)