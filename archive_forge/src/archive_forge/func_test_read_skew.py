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
def test_read_skew(self):
    a = [[1, 0, 0, 0, 0], [0, 10.5, 0, -250.5, 0], [0, 0, 0.015, 0, 0], [0, 250.5, 0, -280, 0], [0, 0, 0, 0, 12]]
    self.check_read(_skew_example, a, (5, 5, 7, 'coordinate', 'real', 'skew-symmetric'))