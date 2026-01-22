import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_delete_last_1_col(self):
    a, q, r = self.generate('Mx1', 'economic')
    q1, r1 = qr_delete(q, r, 0, 1, 'col')
    assert_equal(q1, np.ndarray(shape=(q.shape[0], 0), dtype=q.dtype))
    assert_equal(r1, np.ndarray(shape=(0, 0), dtype=r.dtype))
    a, q, r = self.generate('Mx1', 'full')
    q1, r1 = qr_delete(q, r, 0, 1, 'col')
    assert_unitary(q1)
    assert_(q1.dtype == q.dtype)
    assert_(q1.shape == q.shape)
    assert_equal(r1, np.ndarray(shape=(r.shape[0], 0), dtype=r.dtype))