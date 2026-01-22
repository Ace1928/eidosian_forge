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
def test_input_ranges():
    arr = np.arange(-500, 501, 10, dtype=np.float64)
    bio = BytesIO()
    working_type = np.float32
    work_eps = np.finfo(working_type).eps
    for out_type, offset in itertools.product(IUINT_TYPES, range(-1000, 1000, 100)):
        aw = SlopeInterArrayWriter(arr, out_type)
        aw.to_fileobj(bio)
        arr2 = array_from_file(arr.shape, out_type, bio)
        arr3 = apply_read_scaling(arr2, aw.slope, aw.inter)
        max_miss = np.abs(aw.slope) / working_type(2.0) + work_eps * 10
        abs_err = np.abs(arr - arr3)
        max_err = np.abs(arr) * work_eps + max_miss
        assert np.all(abs_err <= max_err)
        if out_type in UINT_TYPES and 0 in (min(arr), max(arr)):
            assert min(abs_err) == abs_err[arr == 0]
        bio.truncate(0)
        bio.seek(0)