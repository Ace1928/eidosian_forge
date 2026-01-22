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
def test_bad_number_of_coordinate_header_fields(self):
    s = '            %%MatrixMarket matrix coordinate real general\n              5  5  8 999\n                1     1   1.000e+00\n                2     2   1.050e+01\n                3     3   1.500e-02\n                1     4   6.000e+00\n                4     2   2.505e+02\n                4     4  -2.800e+02\n                4     5   3.332e+01\n                5     5   1.200e+01\n            '
    text = textwrap.dedent(s).encode('ascii')
    with pytest.raises(ValueError, match='not of length 3'):
        scipy.io.mmread(io.BytesIO(text))