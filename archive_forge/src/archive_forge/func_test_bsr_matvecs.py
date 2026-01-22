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
def test_bsr_matvecs(self):
    A = bsr_matrix(arange(2 * 3 * 4 * 5).reshape(2 * 4, 3 * 5), blocksize=(4, 5))
    x = arange(A.shape[1] * 6).reshape(-1, 6)
    assert_equal(A * x, A.toarray() @ x)