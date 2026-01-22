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
@sup_complex
def test_from_matrix(self):
    A = matrix([[1, 0, 0], [2, 3, 4], [0, 5, 0], [0, 0, 0]])
    assert_array_equal(self.spcreator(A).todense(), A)
    A = matrix([[1.0 + 3j, 0, 0], [0, 2.0 + 5, 0], [0, 0, 0]])
    assert_array_equal(self.spcreator(A).todense(), A)
    assert_array_equal(self.spcreator(A, dtype='int16').todense(), A.astype('int16'))