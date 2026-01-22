import bz2
import functools
import gzip
import itertools
import os
import tempfile
import threading
import time
import warnings
from io import BytesIO
from os.path import exists
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from packaging.version import Version
from nibabel.testing import (
from ..casting import OK_FLOATS, floor_log2, sctypes, shared_range, type_info
from ..openers import BZ2File, ImageOpener, Opener
from ..optpkg import optional_package
from ..tmpdirs import InTemporaryDirectory
from ..volumeutils import (
def test_a2f_nan2zero():
    ndt = np.dtype(np.float64)
    str_io = BytesIO()
    arr = np.array([[np.nan, 0], [0, np.nan]])
    data_back = write_return(arr, str_io, ndt)
    assert_array_equal(data_back, arr)
    data_back = write_return(arr, str_io, ndt, nan2zero=True)
    assert_array_equal(data_back, arr)
    with np.errstate(invalid='ignore'):
        data_back = write_return(arr, str_io, np.int64, nan2zero=True)
    assert_array_equal(data_back, [[0, 0], [0, 0]])
    with np.errstate(invalid='ignore'):
        data_back = write_return(arr, str_io, np.int64, nan2zero=False)
        assert_array_equal(data_back, arr.astype(np.int64))