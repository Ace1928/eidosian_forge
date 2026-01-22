import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
def test_zerosized_matrix_arg(self):
    a = block_diag([[1, 0], [0, 1]], [[]], [[2, 3], [4, 5], [6, 7]], np.zeros([0, 2], dtype='int32'))
    assert_array_equal(a, [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 2, 3, 0, 0], [0, 0, 4, 5, 0, 0], [0, 0, 6, 7, 0, 0]])