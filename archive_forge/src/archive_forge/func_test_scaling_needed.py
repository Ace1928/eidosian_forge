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
def test_scaling_needed():
    dt_def = [('f', 'i4')]
    arr = np.ones(10, dt_def)
    for t in NUMERIC_TYPES:
        with pytest.raises(WriterError):
            ArrayWriter(arr, t)
        narr = np.ones(10, t)
        with pytest.raises(WriterError):
            ArrayWriter(narr, dt_def)
    assert not ArrayWriter(arr).scaling_needed()
    assert not ArrayWriter(arr, dt_def).scaling_needed()
    for in_t in NUMERIC_TYPES:
        for out_t in NUMERIC_TYPES:
            if np.can_cast(in_t, out_t):
                aw = ArrayWriter(np.ones(10, in_t), out_t)
                assert not aw.scaling_needed()
    for in_t in NUMERIC_TYPES:
        arr = np.ones(10, in_t)
        for out_t in COMPLEX_TYPES:
            assert not ArrayWriter(arr, out_t).scaling_needed()
    for in_t in COMPLEX_TYPES:
        for out_t in FLOAT_TYPES + IUINT_TYPES:
            arr = np.ones(10, in_t)
            with pytest.raises(WriterError):
                ArrayWriter(arr, out_t)
    for in_t in FLOAT_TYPES + IUINT_TYPES:
        arr = np.ones(10, in_t)
        for out_t in FLOAT_TYPES:
            assert not ArrayWriter(arr, out_t).scaling_needed()
    for in_t in FLOAT_TYPES + IUINT_TYPES:
        arr_0 = np.zeros(10, in_t)
        arr_e = []
        for out_t in IUINT_TYPES:
            assert not ArrayWriter(arr_0, out_t).scaling_needed()
            assert not ArrayWriter(arr_e, out_t).scaling_needed()
    for in_t in FLOAT_TYPES:
        arr_nan = np.zeros(10, in_t) + np.nan
        arr_inf = np.zeros(10, in_t) + np.inf
        arr_minf = np.zeros(10, in_t) - np.inf
        arr_mix = np.array([np.nan, np.inf, -np.inf], dtype=in_t)
        for out_t in IUINT_TYPES:
            for arr in (arr_nan, arr_inf, arr_minf, arr_mix):
                assert ArrayWriter(arr, out_t, check_scaling=False).scaling_needed()
                assert not SlopeArrayWriter(arr, out_t).scaling_needed()
                assert not SlopeInterArrayWriter(arr, out_t).scaling_needed()
    for in_t in FLOAT_TYPES:
        arr = np.ones(10, in_t)
        for out_t in IUINT_TYPES:
            assert SlopeArrayWriter(arr, out_t).scaling_needed()
    for in_t in IUINT_TYPES:
        in_info = np.iinfo(in_t)
        in_min, in_max = (in_info.min, in_info.max)
        for out_t in IUINT_TYPES:
            out_info = np.iinfo(out_t)
            out_min, out_max = (out_info.min, out_info.max)
            if in_min >= out_min and in_max <= out_max:
                arr = np.array([in_min, in_max], in_t)
                assert np.can_cast(arr.dtype, out_t)
                assert not ArrayWriter(arr, out_t).scaling_needed()
                continue
            max_min = max(in_min, out_min)
            min_max = min(in_max, out_max)
            arr = np.array([max_min, min_max], in_t)
            assert not ArrayWriter(arr, out_t).scaling_needed()
            assert SlopeInterArrayWriter(arr + 1, out_t).scaling_needed()
            if in_t in INT_TYPES:
                assert SlopeInterArrayWriter(arr - 1, out_t).scaling_needed()