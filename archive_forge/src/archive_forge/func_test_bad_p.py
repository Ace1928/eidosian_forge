import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_bad_p(self):
    a, q, r = self.generate('tall')
    assert_raises(ValueError, qr_delete, q, r, 0, -1)
    assert_raises(ValueError, qr_delete, q, r, 0, -1, 'col')
    assert_raises(ValueError, qr_delete, q, r, 0, 0)
    assert_raises(ValueError, qr_delete, q, r, 0, 0, 'col')
    assert_raises(ValueError, qr_delete, q, r, 3, q.shape[0] - 2)
    assert_raises(ValueError, qr_delete, q, r, 3, r.shape[1] - 2, 'col')