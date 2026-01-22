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
def test_load_with_qid():
    data = b'\n    3 qid:1 1:0.53 2:0.12\n    2 qid:1 1:0.13 2:0.1\n    7 qid:2 1:0.87 2:0.12'
    X, y = load_svmlight_file(BytesIO(data), query_id=False)
    assert_array_equal(y, [3, 2, 7])
    assert_array_equal(X.toarray(), [[0.53, 0.12], [0.13, 0.1], [0.87, 0.12]])
    res1 = load_svmlight_files([BytesIO(data)], query_id=True)
    res2 = load_svmlight_file(BytesIO(data), query_id=True)
    for X, y, qid in (res1, res2):
        assert_array_equal(y, [3, 2, 7])
        assert_array_equal(qid, [1, 1, 2])
        assert_array_equal(X.toarray(), [[0.53, 0.12], [0.13, 0.1], [0.87, 0.12]])