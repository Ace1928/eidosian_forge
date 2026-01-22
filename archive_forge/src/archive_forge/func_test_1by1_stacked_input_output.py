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
def test_1by1_stacked_input_output(self):
    a = self.rng.random([4, 5, 1, 1], dtype=np.float32)
    deta = det(a)
    assert deta.dtype.char == 'd'
    assert deta.shape == (4, 5)
    assert_allclose(deta, np.squeeze(a))
    a = self.rng.random([4, 5, 1, 1], dtype=np.float32) * np.complex64(1j)
    deta = det(a)
    assert deta.dtype.char == 'D'
    assert deta.shape == (4, 5)
    assert_allclose(deta, np.squeeze(a))