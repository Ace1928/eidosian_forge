import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def assert_unitary(a, rtol=None, atol=None, assert_sqr=True):
    if rtol is None:
        rtol = 10.0 ** (-(np.finfo(a.dtype).precision - 2))
    if atol is None:
        atol = 10 * np.finfo(a.dtype).eps
    if assert_sqr:
        assert_(a.shape[0] == a.shape[1], 'unitary matrices must be square')
    aTa = np.dot(a.T.conj(), a)
    assert_allclose(aTa, np.eye(a.shape[1]), rtol=rtol, atol=atol)