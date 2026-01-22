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
def test_empty_write_read(self):
    b = scipy.sparse.coo_matrix((10, 10))
    mmwrite(self.fn, b)
    assert_equal(mminfo(self.fn), (10, 10, 0, 'coordinate', 'real', 'symmetric'))
    a = b.toarray()
    b = mmread(self.fn).toarray()
    assert_array_almost_equal(a, b)