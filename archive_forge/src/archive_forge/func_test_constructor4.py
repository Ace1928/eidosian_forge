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
def test_constructor4(self):
    n = 8
    data = np.ones((n, n, 1), dtype=np.int8)
    indptr = np.array([0, n], dtype=np.int32)
    indices = np.arange(n, dtype=np.int32)
    bsr_matrix((data, indices, indptr), blocksize=(n, 1), copy=False)