import pytest
import numpy as np
from numpy import cos, sin, pi
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hyp_num
from scipy.integrate import (quadrature, romberg, romb, newton_cotes,
from scipy.integrate._quadrature import _cumulative_simpson_unequal_intervals
from scipy.integrate._tanhsinh import _tanhsinh, _pair_cache
from scipy import stats, special as sc
from scipy.optimize._zeros_py import (_ECONVERGED, _ESIGNERR, _ECONVERR,  # noqa: F401
def cumulative_simpson_nd_reference(y, *, x=None, dx=None, initial=None, axis=-1):
    if y.shape[axis] < 3:
        if initial is None:
            return cumulative_trapezoid(y, x=x, dx=dx, axis=axis, initial=None)
        else:
            return initial + cumulative_trapezoid(y, x=x, dx=dx, axis=axis, initial=0)
    y = np.moveaxis(y, axis, -1)
    x = np.moveaxis(x, axis, -1) if np.ndim(x) > 1 else x
    dx = np.moveaxis(dx, axis, -1) if np.ndim(dx) > 1 else dx
    initial = np.moveaxis(initial, axis, -1) if np.ndim(initial) > 1 else initial
    n = y.shape[-1]
    x = dx * np.arange(n) if dx is not None else x
    initial_was_none = initial is None
    initial = 0 if initial_was_none else initial
    x = np.broadcast_to(x, y.shape)
    initial = np.broadcast_to(initial, y.shape[:-1] + (1,))
    z = np.concatenate((y, x, initial), axis=-1)

    def f(z):
        return cumulative_simpson(z[:n], x=z[n:2 * n], initial=z[2 * n:])
    res = np.apply_along_axis(f, -1, z)
    res = res[..., 1:] if initial_was_none else res
    res = np.moveaxis(res, -1, axis)
    return res