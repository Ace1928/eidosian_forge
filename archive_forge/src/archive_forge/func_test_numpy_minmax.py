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
def test_numpy_minmax(self):
    from scipy.sparse import _data
    dat = array([[0, 1, 2], [3, -4, 5], [-6, 7, 9]])
    datsp = self.spcreator(dat)
    if isinstance(datsp, _data._minmax_mixin):
        assert_array_equal(np.min(datsp), np.min(dat))
        assert_array_equal(np.max(datsp), np.max(dat))