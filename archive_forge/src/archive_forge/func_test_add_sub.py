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
def test_add_sub(self):
    self.__arith_init()
    assert_array_equal((self.__Asp + self.__Bsp).toarray(), self.__A + self.__B)
    for x in supported_dtypes:
        with np.errstate(invalid='ignore'):
            A = self.__A.astype(x)
        Asp = self.spcreator(A)
        for y in supported_dtypes:
            if not np.issubdtype(y, np.complexfloating):
                with np.errstate(invalid='ignore'):
                    B = self.__B.real.astype(y)
            else:
                B = self.__B.astype(y)
            Bsp = self.spcreator(B)
            D1 = A + B
            S1 = Asp + Bsp
            assert_equal(S1.dtype, D1.dtype)
            assert_array_equal(S1.toarray(), D1)
            assert_array_equal(Asp + B, D1)
            assert_array_equal(A + Bsp, D1)
            if np.dtype('bool') in [x, y]:
                continue
            D1 = A - B
            S1 = Asp - Bsp
            assert_equal(S1.dtype, D1.dtype)
            assert_array_equal(S1.toarray(), D1)
            assert_array_equal(Asp - B, D1)
            assert_array_equal(A - Bsp, D1)