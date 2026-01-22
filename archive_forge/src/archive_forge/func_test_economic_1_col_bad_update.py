import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_economic_1_col_bad_update(self):
    q = np.eye(5, 3, dtype=self.dtype)
    r = np.eye(3, dtype=self.dtype)
    u = np.array([1, 0, 0, 0, 0], self.dtype)
    assert_raises(linalg.LinAlgError, qr_insert, q, r, u, 0, 'col')