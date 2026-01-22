import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_compress_rowcols(self):
    x = array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert_equal(compress_rowcols(x), [[4, 5], [7, 8]])
    assert_equal(compress_rowcols(x, 0), [[3, 4, 5], [6, 7, 8]])
    assert_equal(compress_rowcols(x, 1), [[1, 2], [4, 5], [7, 8]])
    x = array(x._data, mask=[[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert_equal(compress_rowcols(x), [[0, 2], [6, 8]])
    assert_equal(compress_rowcols(x, 0), [[0, 1, 2], [6, 7, 8]])
    assert_equal(compress_rowcols(x, 1), [[0, 2], [3, 5], [6, 8]])
    x = array(x._data, mask=[[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert_equal(compress_rowcols(x), [[8]])
    assert_equal(compress_rowcols(x, 0), [[6, 7, 8]])
    assert_equal(compress_rowcols(x, 1), [[2], [5], [8]])
    x = array(x._data, mask=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert_equal(compress_rowcols(x).size, 0)
    assert_equal(compress_rowcols(x, 0).size, 0)
    assert_equal(compress_rowcols(x, 1).size, 0)