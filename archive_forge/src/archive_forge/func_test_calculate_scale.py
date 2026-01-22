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
def test_calculate_scale():
    npa = np.array
    SIAW = SlopeInterArrayWriter
    SAW = SlopeArrayWriter
    aw = SIAW(npa([-2, -1], dtype=np.int8), np.uint8)
    assert get_slope_inter(aw) == (1.0, -2.0)
    aw = SAW(npa([-2, -1], dtype=np.int8), np.uint8)
    assert get_slope_inter(aw) == (-1.0, 0.0)
    aw = SAW(npa([-2, 0], dtype=np.int8), np.uint8)
    assert get_slope_inter(aw) == (-1.0, 0.0)
    aw = SAW(npa([-510, 0], dtype=np.int16), np.uint8)
    assert get_slope_inter(aw) == (-2.0, 0.0)
    aw = SAW(npa([-2, 0], dtype=np.float32), np.uint8)
    assert get_slope_inter(aw) != (-1.0, 0.0)
    aw = SIAW(npa([-1, 1], dtype=np.int8), np.uint8)
    assert get_slope_inter(aw) == (1.0, -1.0)
    with pytest.raises(WriterError):
        SAW(npa([-1, 1], dtype=np.int8), np.uint8)
    aw = SIAW(npa([-1, 255], dtype=np.int16), np.uint8)
    slope_inter = get_slope_inter(aw)
    assert slope_inter != (1.0, -1.0)