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
def test_non_unit_stride_2d_indexing(self):
    v0 = np.random.rand(50, 50)
    try:
        v = self.spcreator(v0)[0:25:2, 2:30:3]
    except ValueError:
        raise pytest.skip('feature not implemented')
    assert_array_equal(v.toarray(), v0[0:25:2, 2:30:3])