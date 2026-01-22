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
def assert_rt(data, shape, out_dtype, order='F', in_cast=None, pre_clips=None, inter=0.0, slope=1.0, post_clips=None, nan_fill=None):
    sio = BytesIO()
    to_write = data.reshape(shape)
    backup = to_write.copy()
    nan_positions = np.isnan(to_write)
    have_nans = np.any(nan_positions)
    if have_nans and nan_fill is None and (not out_dtype.type == 'f'):
        raise ValueError('Cannot handle this case')
    _write_data(to_write, sio, out_dtype, order, in_cast, pre_clips, inter, slope, post_clips, nan_fill)
    arr = np.ndarray(shape, out_dtype, buffer=sio.getvalue(), order=order)
    expected = to_write.copy()
    if have_nans and (not nan_fill is None):
        expected[nan_positions] = nan_fill * slope + inter
    assert_array_equal(arr * slope + inter, expected)
    assert_array_equal(to_write, backup)