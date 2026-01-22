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
def test_read_64bit_integer_sparse_symmetric(self):
    a = array([[2 ** 31, -2 ** 63 + 1], [-2 ** 63 + 1, 2 ** 63 - 1]], dtype=np.int64)
    self.check_read(_64bit_integer_sparse_symmetric_example, a, (2, 2, 3, 'coordinate', 'integer', 'symmetric'), dense=False, over32=True, over64=False)