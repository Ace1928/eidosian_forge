import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_sqr_p_col(self):
    a, q, r, u = self.generate('sqr', which='col', p=3)
    for col in range(r.shape[1] + 1):
        q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
        a1 = np.insert(a, np.full(3, col, np.intp), u, 1)
        check_qr(q1, r1, a1, self.rtol, self.atol)