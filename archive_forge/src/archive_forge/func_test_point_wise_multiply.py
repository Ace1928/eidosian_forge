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
def test_point_wise_multiply(self):
    l = lil_matrix((4, 3))
    l[0, 0] = 1
    l[1, 1] = 2
    l[2, 2] = 3
    l[3, 1] = 4
    m = lil_matrix((4, 3))
    m[0, 0] = 1
    m[0, 1] = 2
    m[2, 2] = 3
    m[3, 1] = 4
    m[3, 2] = 4
    assert_array_equal(l.multiply(m).toarray(), m.multiply(l).toarray())
    assert_array_equal(l.multiply(m).toarray(), [[1, 0, 0], [0, 0, 0], [0, 0, 9], [0, 16, 0]])