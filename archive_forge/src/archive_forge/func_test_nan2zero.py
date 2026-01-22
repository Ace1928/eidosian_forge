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
def test_nan2zero():
    arr = np.array([np.nan, 99.0], dtype=np.float32)
    for awt, kwargs in ((ArrayWriter, dict(check_scaling=False)), (SlopeArrayWriter, dict(calc_scale=False)), (SlopeInterArrayWriter, dict(calc_scale=False))):
        aw = awt(arr, np.float32, **kwargs)
        data_back = round_trip(aw)
        assert_array_equal(np.isnan(data_back), [True, False])
        aw = awt(arr, np.float32, nan2zero=True, **kwargs)
        data_back = round_trip(aw)
        assert_array_equal(np.isnan(data_back), [True, False])
        aw = awt(arr, np.int32, **kwargs)
        data_back = round_trip(aw)
        assert_array_equal(data_back, [0, 99])
        aw = awt(arr, np.int32, nan2zero=False, **kwargs)
        data_back = round_trip(aw)
        astype_res = np.array(np.nan).astype(np.int32)
        assert_array_equal(data_back, [astype_res, 99])