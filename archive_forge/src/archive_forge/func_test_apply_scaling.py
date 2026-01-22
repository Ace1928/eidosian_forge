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
def test_apply_scaling():
    arr = np.zeros((3,), dtype=np.int16)
    assert apply_read_scaling(arr) is arr
    assert apply_read_scaling(arr, np.float64(1.0)) is arr
    assert apply_read_scaling(arr, inter=np.float64(0)) is arr
    f32, f64 = (np.float32, np.float64)
    f32_arr = np.zeros((1,), dtype=f32)
    i16_arr = np.zeros((1,), dtype=np.int16)
    assert (f32_arr * 1.0).dtype == np.float32
    assert (f32_arr + 1.0).dtype == np.float32
    want_dtype = np.float64 if NP_2 else np.float32
    assert (f32_arr * f64(1)).dtype == want_dtype
    assert (f32_arr + f64(1)).dtype == want_dtype
    ret = apply_read_scaling(np.float32(0), np.float64(2))
    assert ret.dtype == np.float64
    ret = apply_read_scaling(np.float32(0), inter=np.float64(2))
    assert ret.dtype == np.float64
    big = f32(type_info(f32)['max'])
    assert (i16_arr * big).dtype == np.float32
    nmant_32 = type_info(np.float32)['nmant']
    big_delta = np.float32(2 ** (floor_log2(big) - nmant_32))
    assert (i16_arr * big_delta + big).dtype == np.float32
    assert apply_read_scaling(i16_arr, big).dtype == np.float64
    assert apply_read_scaling(i16_arr, big_delta, big).dtype == np.float64
    assert apply_read_scaling(np.int8(0), f32(-1.0), f32(0.0)).dtype == np.float32
    assert apply_read_scaling(np.int8(0), -1.0, 0.0).dtype == np.float64
    assert apply_read_scaling(np.int8(0), f32(1e+38), f32(0.0)).dtype == np.float64
    assert apply_read_scaling(np.int8(0), f32(-1e+38), f32(0.0)).dtype == np.float64
    assert_dt_equal(apply_read_scaling(i16_arr, 1.0, 1.0).dtype, float)
    assert_dt_equal(apply_read_scaling(np.zeros((1,), dtype=np.int32), 1.0, 1.0).dtype, float)
    assert_dt_equal(apply_read_scaling(np.zeros((1,), dtype=np.int64), 1.0, 1.0).dtype, float)