import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_overwrite_qu_rank_p(self):
    a, q0, r, u = self.generate('sqr', which='col', p=3)
    q = q0.copy('F')
    a1 = np.insert(a, np.zeros(3, np.intp), u, 1)
    q1, r1 = qr_insert(q, r, u, 0, 'col', overwrite_qru=False)
    check_qr(q1, r1, a1, self.rtol, self.atol)
    check_qr(q, r, a, self.rtol, self.atol)
    q2, r2 = qr_insert(q, r, u, 0, 'col', overwrite_qru=True)
    check_qr(q2, r2, a1, self.rtol, self.atol)
    assert_allclose(q2, q, rtol=self.rtol, atol=self.atol)