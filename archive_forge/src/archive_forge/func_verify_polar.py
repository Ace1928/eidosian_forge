import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_, assert_allclose, assert_equal)
from scipy.linalg import polar, eigh
def verify_polar(a):
    product_atol = np.sqrt(np.finfo(float).eps)
    aa = np.asarray(a)
    m, n = aa.shape
    u, p = polar(a, side='right')
    assert_equal(u.shape, (m, n))
    assert_equal(p.shape, (n, n))
    assert_allclose(u.dot(p), a, atol=product_atol)
    if m >= n:
        assert_allclose(u.conj().T.dot(u), np.eye(n), atol=1e-15)
    else:
        assert_allclose(u.dot(u.conj().T), np.eye(m), atol=1e-15)
    assert_allclose(p.conj().T, p)
    evals = eigh(p, eigvals_only=True)
    nonzero_evals = evals[abs(evals) > 1e-14]
    assert_((nonzero_evals >= 0).all())
    u, p = polar(a, side='left')
    assert_equal(u.shape, (m, n))
    assert_equal(p.shape, (m, m))
    assert_allclose(p.dot(u), a, atol=product_atol)
    if m >= n:
        assert_allclose(u.conj().T.dot(u), np.eye(n), atol=1e-15)
    else:
        assert_allclose(u.dot(u.conj().T), np.eye(m), atol=1e-15)
    assert_allclose(p.conj().T, p)
    evals = eigh(p, eigvals_only=True)
    nonzero_evals = evals[abs(evals) > 1e-14]
    assert_((nonzero_evals >= 0).all())