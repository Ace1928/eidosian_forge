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
def test_a2f_order():
    ndt = np.dtype(np.float64)
    arr = np.array([0.0, 1.0, 2.0])
    str_io = BytesIO()
    data_back = write_return(arr, str_io, ndt, order='C')
    assert_array_equal(data_back, [0.0, 1.0, 2.0])
    arr = np.array([[0.0, 1.0], [2.0, 3.0]])
    data_back = write_return(arr, str_io, ndt, order='F')
    assert_array_equal(data_back, arr)
    data_back = write_return(arr, str_io, ndt, order='C')
    assert_array_equal(data_back, arr.T)