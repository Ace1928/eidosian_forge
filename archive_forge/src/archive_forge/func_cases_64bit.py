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
def cases_64bit():
    TEST_CLASSES = [TestBSR, TestCOO, TestCSC, TestCSR, TestDIA, TestDOK, TestLIL]
    SKIP_TESTS = {'test_expm': 'expm for 64-bit indices not available', 'test_inv': 'linsolve for 64-bit indices not available', 'test_solve': 'linsolve for 64-bit indices not available', 'test_scalar_idx_dtype': 'test implemented in base class', 'test_large_dimensions_reshape': 'test actually requires 64-bit to work', 'test_constructor_smallcol': 'test verifies int32 indexes', 'test_constructor_largecol': 'test verifies int64 indexes', 'test_tocoo_tocsr_tocsc_gh19245': 'test verifies int32 indexes'}
    for cls in TEST_CLASSES:
        for method_name in sorted(dir(cls)):
            method = getattr(cls, method_name)
            if method_name.startswith('test_') and (not getattr(method, 'slow', False)):
                marks = []
                msg = SKIP_TESTS.get(method_name)
                if bool(msg):
                    marks += [pytest.mark.skip(reason=msg)]
                if _pep440.parse(pytest.__version__) >= _pep440.Version('3.6.0'):
                    markers = getattr(method, 'pytestmark', [])
                    for mark in markers:
                        if mark.name in ('skipif', 'skip', 'xfail', 'xslow'):
                            marks.append(mark)
                else:
                    for mname in ['skipif', 'skip', 'xfail', 'xslow']:
                        if hasattr(method, mname):
                            marks += [getattr(method, mname)]
                yield pytest.param(cls, method_name, marks=marks)