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
def test_gh13634_non_skew_symmetric_int(self):
    a = scipy.sparse.csr_matrix([[1, 2], [-2, 99]], dtype=np.int32)
    self.check_exact(a, (2, 2, 4, 'coordinate', 'integer', 'general'))