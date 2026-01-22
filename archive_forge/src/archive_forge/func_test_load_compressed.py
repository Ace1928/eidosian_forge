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
def test_load_compressed():
    X, y = _load_svmlight_local_test_file(datafile)
    with NamedTemporaryFile(prefix='sklearn-test', suffix='.gz') as tmp:
        tmp.close()
        with _svmlight_local_test_file_path(datafile).open('rb') as f:
            with gzip.open(tmp.name, 'wb') as fh_out:
                shutil.copyfileobj(f, fh_out)
        Xgz, ygz = load_svmlight_file(tmp.name)
        os.remove(tmp.name)
    assert_array_almost_equal(X.toarray(), Xgz.toarray())
    assert_array_almost_equal(y, ygz)
    with NamedTemporaryFile(prefix='sklearn-test', suffix='.bz2') as tmp:
        tmp.close()
        with _svmlight_local_test_file_path(datafile).open('rb') as f:
            with BZ2File(tmp.name, 'wb') as fh_out:
                shutil.copyfileobj(f, fh_out)
        Xbz, ybz = load_svmlight_file(tmp.name)
        os.remove(tmp.name)
    assert_array_almost_equal(X.toarray(), Xbz.toarray())
    assert_array_almost_equal(y, ybz)