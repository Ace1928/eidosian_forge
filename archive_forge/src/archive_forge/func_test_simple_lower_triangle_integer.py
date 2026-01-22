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
@pytest.mark.parametrize('typeval, dtype', parametrize_args)
def test_simple_lower_triangle_integer(self, typeval, dtype):
    self.check_exact(scipy.sparse.csr_matrix([[0, 0], [1, 0]], dtype=dtype), (2, 2, 1, 'coordinate', typeval, 'general'))