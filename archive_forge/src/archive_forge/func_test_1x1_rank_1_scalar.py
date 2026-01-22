import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_1x1_rank_1_scalar(self):
    a, q, r, u, v = self.generate('1x1')
    assert_raises(ValueError, qr_update, q[0, 0], r, u, v)
    assert_raises(ValueError, qr_update, q, r[0, 0], u, v)
    assert_raises(ValueError, qr_update, q, r, u[0], v)
    assert_raises(ValueError, qr_update, q, r, u, v[0])