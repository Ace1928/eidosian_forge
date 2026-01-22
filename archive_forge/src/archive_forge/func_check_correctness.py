import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
@staticmethod
def check_correctness(S, bc_start='not-a-knot', bc_end='not-a-knot', tol=1e-14):
    """Check that spline coefficients satisfy the continuity and boundary
        conditions."""
    x = S.x
    c = S.c
    dx = np.diff(x)
    dx = dx.reshape([dx.shape[0]] + [1] * (c.ndim - 2))
    dxi = dx[:-1]
    assert_allclose(c[3, 1:], c[0, :-1] * dxi ** 3 + c[1, :-1] * dxi ** 2 + c[2, :-1] * dxi + c[3, :-1], rtol=tol, atol=tol)
    assert_allclose(c[2, 1:], 3 * c[0, :-1] * dxi ** 2 + 2 * c[1, :-1] * dxi + c[2, :-1], rtol=tol, atol=tol)
    assert_allclose(c[1, 1:], 3 * c[0, :-1] * dxi + c[1, :-1], rtol=tol, atol=tol)
    if x.size == 3 and bc_start == 'not-a-knot' and (bc_end == 'not-a-knot'):
        assert_allclose(c[0], 0, rtol=tol, atol=tol)
        return
    if bc_start == 'periodic':
        assert_allclose(S(x[0], 0), S(x[-1], 0), rtol=tol, atol=tol)
        assert_allclose(S(x[0], 1), S(x[-1], 1), rtol=tol, atol=tol)
        assert_allclose(S(x[0], 2), S(x[-1], 2), rtol=tol, atol=tol)
        return
    if bc_start == 'not-a-knot':
        if x.size == 2:
            slope = (S(x[1]) - S(x[0])) / dx[0]
            assert_allclose(S(x[0], 1), slope, rtol=tol, atol=tol)
        else:
            assert_allclose(c[0, 0], c[0, 1], rtol=tol, atol=tol)
    elif bc_start == 'clamped':
        assert_allclose(S(x[0], 1), 0, rtol=tol, atol=tol)
    elif bc_start == 'natural':
        assert_allclose(S(x[0], 2), 0, rtol=tol, atol=tol)
    else:
        order, value = bc_start
        assert_allclose(S(x[0], order), value, rtol=tol, atol=tol)
    if bc_end == 'not-a-knot':
        if x.size == 2:
            slope = (S(x[1]) - S(x[0])) / dx[0]
            assert_allclose(S(x[1], 1), slope, rtol=tol, atol=tol)
        else:
            assert_allclose(c[0, -1], c[0, -2], rtol=tol, atol=tol)
    elif bc_end == 'clamped':
        assert_allclose(S(x[-1], 1), 0, rtol=tol, atol=tol)
    elif bc_end == 'natural':
        assert_allclose(S(x[-1], 2), 0, rtol=2 * tol, atol=2 * tol)
    else:
        order, value = bc_end
        assert_allclose(S(x[-1], order), value, rtol=tol, atol=tol)