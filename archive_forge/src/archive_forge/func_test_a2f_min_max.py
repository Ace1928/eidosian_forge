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
def test_a2f_min_max():
    str_io = BytesIO()
    for in_dt in (np.float32, np.int8):
        for out_dt in (np.float32, np.int8):
            arr = np.arange(4, dtype=in_dt)
            with np.errstate(invalid='ignore'):
                data_back = write_return(arr, str_io, out_dt, 0, 0, 1, 1)
            assert_array_equal(data_back, [1, 1, 2, 3])
            with np.errstate(invalid='ignore'):
                data_back = write_return(arr, str_io, out_dt, 0, 0, 1, None, 2)
            assert_array_equal(data_back, [0, 1, 2, 2])
            data_back = write_return(arr, str_io, out_dt, 0, 0, 1, 1, 2)
            assert_array_equal(data_back, [1, 1, 2, 2])
    arr = np.arange(4, dtype=np.float32)
    data_back = write_return(arr, str_io, int, 0, -1, 0.5, 1, 2)
    assert_array_equal(data_back * 0.5 - 1, [1, 1, 2, 2])
    data_back = write_return(arr, str_io, int, 0, 1, -0.5, 1, 2)
    assert_array_equal(data_back * -0.5 + 1, [1, 1, 2, 2])
    arr = np.arange(4, dtype=np.complex64) + 100j
    with suppress_warnings():
        data_back = write_return(arr, str_io, out_dt, 0, 0, 1, 1, 2)
    assert_array_equal(data_back, [1, 1, 2, 2])