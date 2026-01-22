from tempfile import mkdtemp
import os
import io
import shutil
import textwrap
import numpy as np
from numpy import array, transpose, pi
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
import scipy.sparse
import scipy.io._mmio
import scipy.io._fast_matrix_market as fmm
def test_complex_write_read(self):
    I = array([0, 0, 1, 2, 3, 3, 3, 4])
    J = array([0, 3, 1, 2, 1, 3, 4, 4])
    V = array([1.0 + 3j, 6.0 + 2j, 10.5 + 0.9j, 0.015 + -4.4j, 250.5 + 0j, -280.0 + 5j, 33.32 + 6.4j, 12.0 + 0.8j])
    b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))
    mmwrite(self.fn, b)
    assert_equal(mminfo(self.fn), (5, 5, 8, 'coordinate', 'complex', 'general'))
    a = b.toarray()
    b = mmread(self.fn).toarray()
    assert_array_almost_equal(a, b)