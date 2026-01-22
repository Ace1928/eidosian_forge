import os
import numpy as np
import h5py
from .common import ut, TestCase
@ut.skipIf(not os.getenv('H5PY_TEST_CHECK_FILTERS'), 'H5PY_TEST_CHECK_FILTERS not set')
def test_filters_available():
    assert 'gzip' in h5py.filters.decode
    assert 'gzip' in h5py.filters.encode
    assert 'lzf' in h5py.filters.decode
    assert 'lzf' in h5py.filters.encode