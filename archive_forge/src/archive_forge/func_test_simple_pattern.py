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
def test_simple_pattern(self):
    a = scipy.sparse.csr_matrix([[0, 1.5], [3.0, 2.5]])
    p = np.zeros_like(a.toarray())
    p[a.toarray() > 0] = 1
    info = (2, 2, 3, 'coordinate', 'pattern', 'general')
    mmwrite(self.fn, a, field='pattern')
    assert_equal(mminfo(self.fn), info)
    b = mmread(self.fn)
    assert_array_almost_equal(p, b.toarray())