import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_u_exactly_in_span_q(self):
    q = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], self.dtype)
    r = np.array([[1, 0], [0, 1]], self.dtype)
    u = np.array([0, 0, 0, -1], self.dtype)
    v = np.array([1, 2], self.dtype)
    q1, r1 = qr_update(q, r, u, v)
    a1 = np.dot(q, r) + np.outer(u, v.conj())
    check_qr(q1, r1, a1, self.rtol, self.atol, False)