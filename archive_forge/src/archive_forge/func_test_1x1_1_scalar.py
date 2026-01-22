import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_1x1_1_scalar(self):
    a, q, r, u = self.generate('1x1', which='row')
    assert_raises(ValueError, qr_insert, q[0, 0], r, u, 0, 'row')
    assert_raises(ValueError, qr_insert, q, r[0, 0], u, 0, 'row')
    assert_raises(ValueError, qr_insert, q, r, u[0], 0, 'row')
    assert_raises(ValueError, qr_insert, q[0, 0], r, u, 0, 'col')
    assert_raises(ValueError, qr_insert, q, r[0, 0], u, 0, 'col')
    assert_raises(ValueError, qr_insert, q, r, u[0], 0, 'col')