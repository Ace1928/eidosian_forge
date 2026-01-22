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
def test_astype_immutable(self):
    D = array([[2.0 + 3j, 0, 0], [0, 4.0 + 5j, 0], [0, 0, 0]])
    S = self.spcreator(D)
    if hasattr(S, 'data'):
        S.data.flags.writeable = False
    if hasattr(S, 'indptr'):
        S.indptr.flags.writeable = False
    if hasattr(S, 'indices'):
        S.indices.flags.writeable = False
    for x in supported_dtypes:
        D_casted = D.astype(x)
        S_casted = S.astype(x)
        assert_equal(S_casted.dtype, D_casted.dtype)