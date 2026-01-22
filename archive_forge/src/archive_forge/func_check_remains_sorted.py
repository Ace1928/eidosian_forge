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
@contextlib.contextmanager
def check_remains_sorted(X):
    """Checks that sorted indices property is retained through an operation
    """
    if not hasattr(X, 'has_sorted_indices') or not X.has_sorted_indices:
        yield
        return
    yield
    indices = X.indices.copy()
    X.has_sorted_indices = False
    X.sort_indices()
    assert_array_equal(indices, X.indices, 'Expected sorted indices, found unsorted')