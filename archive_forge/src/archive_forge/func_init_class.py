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
@classmethod
def init_class(cls):
    cls.dat = array([[1, 0, 0, 2], [3, 0, 1, 0], [0, 2, 0, 0]], 'd')
    cls.datsp = cls.spcreator(cls.dat)
    cls.checked_dtypes = set(supported_dtypes).union(cls.math_dtypes)
    cls.dat_dtypes = {}
    cls.datsp_dtypes = {}
    for dtype in cls.checked_dtypes:
        cls.dat_dtypes[dtype] = cls.dat.astype(dtype)
        cls.datsp_dtypes[dtype] = cls.spcreator(cls.dat.astype(dtype))
    assert_equal(cls.dat, cls.dat_dtypes[np.float64])
    assert_equal(cls.datsp.toarray(), cls.datsp_dtypes[np.float64].toarray())