import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def check_qr(q, r, a, rtol, atol, assert_sqr=True):
    assert_unitary(q, rtol, atol, assert_sqr)
    assert_upper_tri(r, rtol, atol)
    assert_allclose(q.dot(r), a, rtol=rtol, atol=atol)