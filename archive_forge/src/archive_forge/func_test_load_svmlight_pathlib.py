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
def test_load_svmlight_pathlib():
    data_path = _svmlight_local_test_file_path(datafile)
    X1, y1 = load_svmlight_file(str(data_path))
    X2, y2 = load_svmlight_file(data_path)
    assert_allclose(X1.data, X2.data)
    assert_allclose(y1, y2)