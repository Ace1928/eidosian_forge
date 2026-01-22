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
def test_decorator_maxval_random(self):

    @with_64bit_maxval_limit(random=True)
    def check(mat_cls):
        seen_32 = False
        seen_64 = False
        for k in range(100):
            m = self._create_some_matrix(mat_cls, 9, 9)
            seen_32 = seen_32 or self._compare_index_dtype(m, np.int32)
            seen_64 = seen_64 or self._compare_index_dtype(m, np.int64)
            if seen_32 and seen_64:
                break
        else:
            raise AssertionError('both 32 and 64 bit indices not seen')
    for mat_cls in self.MAT_CLASSES:
        check(mat_cls)