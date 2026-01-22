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
def new_get_index_dtype(arrays=(), maxval=None, check_contents=False):
    dtype = np.int32
    if maxval is not None:
        if maxval > maxval_limit:
            dtype = np.int64
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.dtype > np.int32:
            if check_contents:
                if arr.size == 0:
                    continue
                elif np.issubdtype(arr.dtype, np.integer):
                    maxval = arr.max()
                    minval = arr.min()
                    if minval >= -maxval_limit and maxval <= maxval_limit:
                        continue
            dtype = np.int64
    return dtype