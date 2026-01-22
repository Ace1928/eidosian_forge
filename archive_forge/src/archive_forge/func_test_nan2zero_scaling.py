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
def test_nan2zero_scaling():
    for awt, in_dt, out_dt, sign in itertools.product((SlopeArrayWriter, SlopeInterArrayWriter), FLOAT_TYPES, IUINT_TYPES, (-1, 1)):
        in_info = type_info(in_dt)
        out_info = type_info(out_dt)
        if in_info['min'] == 0 and sign == -1:
            continue
        mx = min(in_info['max'], out_info['max'] * 2.0, 2 ** 32)
        vals = [np.nan] + [100, mx]
        nan_arr = np.array(vals, dtype=in_dt) * sign
        nan_arr_0 = np.array([0] + vals, dtype=in_dt) * sign
        zero_arr = np.nan_to_num(nan_arr)
        nan_aw = awt(nan_arr, out_dt, nan2zero=True)
        back_nan = round_trip(nan_aw) * float(sign)
        nan_0_aw = awt(nan_arr_0, out_dt, nan2zero=True)
        back_nan_0 = round_trip(nan_0_aw) * float(sign)
        zero_aw = awt(zero_arr, out_dt, nan2zero=True)
        back_zero = round_trip(zero_aw) * float(sign)
        assert np.allclose(back_nan[1:], back_zero[1:])
        assert_array_equal(back_nan[1:], back_nan_0[2:])
        assert np.abs(back_nan[0] - back_zero[0]) < 0.01
        assert back_nan_0[0] == back_nan_0[1]