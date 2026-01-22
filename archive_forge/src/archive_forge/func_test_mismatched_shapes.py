import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_mismatched_shapes(self):
    a, q, r, u, v = self.generate('tall')
    assert_raises(ValueError, qr_update, q, r[1:], u, v)
    assert_raises(ValueError, qr_update, q[:-2], r, u, v)
    assert_raises(ValueError, qr_update, q, r, u[1:], v)
    assert_raises(ValueError, qr_update, q, r, u, v[1:])