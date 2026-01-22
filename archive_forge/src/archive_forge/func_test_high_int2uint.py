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
def test_high_int2uint():
    arr = np.array([2 ** 63], dtype=np.uint64)
    out_type = np.int64
    aw = SlopeInterArrayWriter(arr, out_type)
    assert aw.inter == 2 ** 63