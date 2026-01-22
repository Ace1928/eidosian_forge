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
def test_fancy_assign_slice(self):
    np.random.seed(1234)
    D = asmatrix(np.random.rand(5, 7))
    S = self.spcreator(D)
    I = [1, 2, 3, 3, 4, 2]
    J = [5, 6, 3, 2, 3, 1]
    I_bad = [ii + 5 for ii in I]
    J_bad = [jj + 7 for jj in J]
    C1 = [1, 2, 3, 4, 5, 6, 7]
    C2 = np.arange(5)[:, None]
    assert_raises(IndexError, S.__setitem__, (I_bad, slice(None)), C1)
    assert_raises(IndexError, S.__setitem__, (slice(None), J_bad), C2)