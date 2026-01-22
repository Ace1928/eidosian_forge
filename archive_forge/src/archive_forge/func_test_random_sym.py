import itertools
import warnings
import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
from numpy.random import random
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,
from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory, IS_MUSL
from scipy.linalg.blas import HAS_ILP64
from scipy._lib.deprecation import _NoValue
def test_random_sym(self):
    n = 20
    a = random([n, n])
    for i in range(n):
        a[i, i] = abs(20 * (0.1 + a[i, i]))
        for j in range(i):
            a[i, j] = a[j, i]
    for i in range(4):
        b = random([n])
        x = solve(a, b, assume_a='pos')
        assert_array_almost_equal(dot(a, x), b)