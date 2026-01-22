import itertools
from io import BytesIO
from platform import machine, python_compiler
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..arraywriters import (
from ..casting import int_abs, sctypes, shared_range, type_info
from ..testing import assert_allclose_safely, suppress_warnings
from ..volumeutils import _dt_min_max, apply_read_scaling, array_from_file
def rt_err_estimate(arr_t, out_dtype, slope, inter):
    slope = 1 if slope is None else slope
    inter = 1 if inter is None else inter
    max_int_miss = slope / 2.0
    flt_there = (arr_t - inter) / slope
    flt_back = flt_there.astype(out_dtype) * slope + inter
    max_flt_miss = np.abs(arr_t - flt_back).max()
    return max_int_miss + max_flt_miss