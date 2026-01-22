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
def test_size_zero_matrix_arithmetic(self):
    mat = array([])
    a = mat.reshape((0, 0))
    b = mat.reshape((0, 1))
    c = mat.reshape((0, 5))
    d = mat.reshape((1, 0))
    e = mat.reshape((5, 0))
    f = np.ones([5, 5])
    asp = self.spcreator(a)
    bsp = self.spcreator(b)
    csp = self.spcreator(c)
    dsp = self.spcreator(d)
    esp = self.spcreator(e)
    fsp = self.spcreator(f)
    assert_array_equal(asp.dot(asp).toarray(), np.dot(a, a))
    assert_array_equal(bsp.dot(dsp).toarray(), np.dot(b, d))
    assert_array_equal(dsp.dot(bsp).toarray(), np.dot(d, b))
    assert_array_equal(csp.dot(esp).toarray(), np.dot(c, e))
    assert_array_equal(csp.dot(fsp).toarray(), np.dot(c, f))
    assert_array_equal(esp.dot(csp).toarray(), np.dot(e, c))
    assert_array_equal(dsp.dot(csp).toarray(), np.dot(d, c))
    assert_array_equal(fsp.dot(esp).toarray(), np.dot(f, e))
    assert_raises(ValueError, dsp.dot, e)
    assert_raises(ValueError, asp.dot, d)
    assert_array_equal(asp.multiply(asp).toarray(), np.multiply(a, a))
    assert_array_equal(bsp.multiply(bsp).toarray(), np.multiply(b, b))
    assert_array_equal(dsp.multiply(dsp).toarray(), np.multiply(d, d))
    assert_array_equal(asp.multiply(a).toarray(), np.multiply(a, a))
    assert_array_equal(bsp.multiply(b).toarray(), np.multiply(b, b))
    assert_array_equal(dsp.multiply(d).toarray(), np.multiply(d, d))
    assert_array_equal(asp.multiply(6).toarray(), np.multiply(a, 6))
    assert_array_equal(bsp.multiply(6).toarray(), np.multiply(b, 6))
    assert_array_equal(dsp.multiply(6).toarray(), np.multiply(d, 6))
    assert_raises(ValueError, asp.multiply, c)
    assert_raises(ValueError, esp.multiply, c)
    assert_array_equal(asp.__add__(asp).toarray(), a.__add__(a))
    assert_array_equal(bsp.__add__(bsp).toarray(), b.__add__(b))
    assert_array_equal(dsp.__add__(dsp).toarray(), d.__add__(d))
    assert_raises(ValueError, asp.__add__, dsp)
    assert_raises(ValueError, bsp.__add__, asp)