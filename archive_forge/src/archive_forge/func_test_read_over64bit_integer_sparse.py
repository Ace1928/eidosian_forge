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
def test_read_over64bit_integer_sparse(self):
    self.check_read(_over64bit_integer_sparse_example, None, (2, 2, 2, 'coordinate', 'integer', 'symmetric'), dense=False, over32=True, over64=True)