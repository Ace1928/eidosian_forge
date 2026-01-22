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
def test_asfptype(self):
    A = self.spcreator(arange(6, dtype='int32').reshape(2, 3))
    assert_equal(A.dtype, np.dtype('int32'))
    assert_equal(A.asfptype().dtype, np.dtype('float64'))
    assert_equal(A.asfptype().format, A.format)
    assert_equal(A.astype('int16').asfptype().dtype, np.dtype('float32'))
    assert_equal(A.astype('complex128').asfptype().dtype, np.dtype('complex128'))
    B = A.asfptype()
    C = B.asfptype()
    assert_(B is C)