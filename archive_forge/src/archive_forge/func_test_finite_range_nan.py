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
def test_finite_range_nan():
    for in_arr, res in (([[-1, 0, 1], [np.inf, np.nan, -np.inf]], (-1, 1)), (np.array([[-1, 0, 1], [np.inf, np.nan, -np.inf]]), (-1, 1)), ([[np.nan], [np.nan]], (np.inf, -np.inf)), (np.zeros((3, 4, 5)) + np.nan, (np.inf, -np.inf)), ([[-np.inf], [np.inf]], (np.inf, -np.inf)), (np.zeros((3, 4, 5)) + np.inf, (np.inf, -np.inf)), ([[np.nan, -1, 2], [-2, np.nan, 1]], (-2, 2)), ([[np.nan, -np.inf, 2], [-2, np.nan, np.inf]], (-2, 2)), ([[-np.inf, 2], [np.nan, 1]], (1, 2)), ([[np.nan, -np.inf, 2], [-2, np.nan, np.inf]], (-2, 2)), ([np.nan], (np.inf, -np.inf)), ([np.inf], (np.inf, -np.inf)), ([-np.inf], (np.inf, -np.inf)), ([np.inf, 1], (1, 1)), ([-np.inf, 1], (1, 1)), ([[], []], (np.inf, -np.inf)), (np.array([[-3, 0, 1], [2, -1, 4]], dtype=int), (-3, 4)), (np.array([[1, 0, 1], [2, 3, 4]], dtype=np.uint), (0, 4)), ([0.0, 1, 2, 3], (0, 3)), ([[np.nan, -1 - 100j, 2], [-2, np.nan, 1 + 100j]], (-2, 2)), ([[np.nan, -1, 2 - 100j], [-2 + 100j, np.nan, 1]], (-2 + 100j, 2 - 100j))):
        for awt, kwargs in ((ArrayWriter, dict(check_scaling=False)), (SlopeArrayWriter, {}), (SlopeArrayWriter, dict(calc_scale=False)), (SlopeInterArrayWriter, {}), (SlopeInterArrayWriter, dict(calc_scale=False))):
            for out_type in NUMERIC_TYPES:
                has_nan = np.any(np.isnan(in_arr))
                try:
                    aw = awt(in_arr, out_type, **kwargs)
                except WriterError:
                    continue
                assert aw.has_nan == has_nan
                assert aw.finite_range() == res
                aw = awt(in_arr, out_type, **kwargs)
                assert aw.finite_range() == res
                assert aw.has_nan == has_nan
                in_arr = np.array(in_arr)
                if in_arr.dtype.kind == 'f':
                    c_arr = in_arr.astype(np.complex128)
                    try:
                        aw = awt(c_arr, out_type, **kwargs)
                    except WriterError:
                        continue
                    aw = awt(c_arr, out_type, **kwargs)
                    assert aw.has_nan == has_nan
                    assert aw.finite_range() == res
            a = np.array([[1.0, 0, 1], [2, 3, 4]]).view([('f1', 'f')])
            aw = awt(a, a.dtype, **kwargs)
            with pytest.raises(TypeError):
                aw.finite_range()
            assert not aw.has_nan