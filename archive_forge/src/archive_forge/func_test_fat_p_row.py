import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_fat_p_row(self):
    a, q, r = self.generate('fat')
    for ndel in range(2, 6):
        for row in range(a.shape[0] - ndel):
            q1, r1 = qr_delete(q, r, row, ndel, overwrite_qr=False)
            a1 = np.delete(a, slice(row, row + ndel), 0)
            check_qr(q1, r1, a1, self.rtol, self.atol)