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
def test_dump_concise():
    one = 1
    two = 2.1
    three = 3.01
    exact = 1.000000000000001
    almost = 1.0
    X = [[one, two, three, exact, almost], [1000000000.0, 2e+18, 3e+27, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    y = [one, two, three, exact, almost]
    f = BytesIO()
    dump_svmlight_file(X, y, f)
    f.seek(0)
    assert f.readline() == b'1 0:1 1:2.1 2:3.01 3:1.000000000000001 4:1\n'
    assert f.readline() == b'2.1 0:1000000000 1:2e+18 2:3e+27\n'
    assert f.readline() == b'3.01 \n'
    assert f.readline() == b'1.000000000000001 \n'
    assert f.readline() == b'1 \n'
    f.seek(0)
    X2, y2 = load_svmlight_file(f)
    assert_array_almost_equal(X, X2.toarray())
    assert_array_almost_equal(y, y2)