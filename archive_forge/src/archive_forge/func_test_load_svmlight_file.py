import gzip
import os
import shutil
from bz2 import BZ2File
from importlib import resources
from io import BytesIO
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
import scipy.sparse as sp
import sklearn
from sklearn.datasets import dump_svmlight_file, load_svmlight_file, load_svmlight_files
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_load_svmlight_file():
    X, y = _load_svmlight_local_test_file(datafile)
    assert X.indptr.shape[0] == 7
    assert X.shape[0] == 6
    assert X.shape[1] == 21
    assert y.shape[0] == 6
    for i, j, val in ((0, 2, 2.5), (0, 10, -5.2), (0, 15, 1.5), (1, 5, 1.0), (1, 12, -3), (2, 20, 27)):
        assert X[i, j] == val
    assert X[0, 3] == 0
    assert X[0, 5] == 0
    assert X[1, 8] == 0
    assert X[1, 16] == 0
    assert X[2, 18] == 0
    X[0, 2] *= 2
    assert X[0, 2] == 5
    assert_array_equal(y, [1, 2, 3, 4, 1, 2])