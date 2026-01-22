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
def test_a2f_big_scalers():
    info = type_info(np.float32)
    arr = np.array([info['min'], 0, info['max']], dtype=np.float32)
    str_io = BytesIO()
    with suppress_warnings():
        array_to_file(arr, str_io, np.int8, intercept=np.float32(2 ** 120), nan2zero=False)
    data_back = array_from_file(arr.shape, np.int8, str_io)
    assert_array_equal(data_back, [-128, -128, 127])
    str_io.seek(0)
    array_to_file(arr, str_io, np.int8, mn=info['min'], mx=info['max'], intercept=np.float32(2 ** 120), nan2zero=False)
    data_back = array_from_file(arr.shape, np.int8, str_io)
    assert_array_equal(data_back, [-128, -128, 127])
    str_io.seek(0)
    with suppress_warnings():
        array_to_file(arr, str_io, np.int8, divslope=np.float32(0.5))
        data_back = array_from_file(arr.shape, np.int8, str_io)
    assert_array_equal(data_back, [-128, 0, 127])
    str_io.seek(0)
    array_to_file(arr, str_io, np.int8, mn=info['min'], mx=info['max'], divslope=np.float32(0.5))
    data_back = array_from_file(arr.shape, np.int8, str_io)
    assert_array_equal(data_back, [-128, 0, 127])