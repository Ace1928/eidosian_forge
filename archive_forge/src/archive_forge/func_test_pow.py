import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
def test_pow(self):
    A = array([[1, 0, 2, 0], [0, 3, 4, 0], [0, 5, 0, 0], [0, 6, 7, 8]])
    B = self.spcreator(A)
    for exponent in [0, 1, 2, 3]:
        ret_sp = B ** exponent
        ret_np = np.linalg.matrix_power(A, exponent)
        assert_array_equal(ret_sp.toarray(), ret_np)
        assert_equal(ret_sp.dtype, ret_np.dtype)
    for exponent in [-1, 2.2, 1 + 3j]:
        assert_raises(ValueError, B.__pow__, exponent)
    B = self.spcreator(A[:3, :])
    assert_raises(TypeError, B.__pow__, 1)