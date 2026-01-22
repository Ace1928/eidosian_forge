import sys
import platform
import pytest
import numpy as np
import numpy.core._multiarray_umath as ncu
from numpy.testing import (
def test_cabs_inf_nan(self):
    x, y = ([], [])
    x.append(np.nan)
    y.append(np.nan)
    check_real_value(np.abs, np.nan, np.nan, np.nan)
    x.append(np.nan)
    y.append(-np.nan)
    check_real_value(np.abs, -np.nan, np.nan, np.nan)
    x.append(np.inf)
    y.append(np.nan)
    check_real_value(np.abs, np.inf, np.nan, np.inf)
    x.append(-np.inf)
    y.append(np.nan)
    check_real_value(np.abs, -np.inf, np.nan, np.inf)

    def f(a):
        return np.abs(np.conj(a))

    def g(a, b):
        return np.abs(complex(a, b))
    xa = np.array(x, dtype=complex)
    assert len(xa) == len(x) == len(y)
    for xi, yi in zip(x, y):
        ref = g(xi, yi)
        check_real_value(f, xi, yi, ref)