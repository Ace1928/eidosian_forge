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
def test_gh18123(tmp_path):
    lines = [' %%MatrixMarket matrix coordinate real general\n', '5 5 3\n', '2 3 1.0\n', '3 4 2.0\n', '3 5 3.0\n']
    test_file = tmp_path / 'test.mtx'
    with open(test_file, 'w') as f:
        f.writelines(lines)
    mmread(test_file)