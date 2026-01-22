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
def test_a2f_nan2zero_scaling():
    bio = BytesIO()
    for in_dt, out_dt, zero_in, inter in itertools.product(FLOAT_TYPES, IUINT_TYPES, (True, False), (0, -100)):
        in_info = np.finfo(in_dt)
        out_info = np.iinfo(out_dt)
        mx = min(in_info.max, out_info.max * 2.0, 2 ** 32) + inter
        mn = 0 if zero_in or inter else 100
        vals = [np.nan] + [mn, mx]
        nan_arr = np.array(vals, dtype=in_dt)
        zero_arr = np.nan_to_num(nan_arr)
        with np.errstate(invalid='ignore'):
            back_nan = write_return(nan_arr, bio, np.int64, intercept=inter)
            back_zero = write_return(zero_arr, bio, np.int64, intercept=inter)
        assert_array_equal(back_nan, back_zero)