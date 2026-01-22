import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..casting import sctypes, type_info
from ..testing import suppress_warnings
from ..volumeutils import apply_read_scaling, array_from_file, array_to_file, finite_range
from .test_volumeutils import _calculate_scale
def check_int_a2f(in_type, out_type):
    big_floater = sctypes['float'][-1]
    info = type_info(in_type)
    this_min, this_max = (info['min'], info['max'])
    if not in_type in sctypes['complex']:
        data = np.array([this_min, this_max], in_type)
        if not np.all(np.isfinite(data)):
            if DEBUG:
                print(f'Hit PPC max -> inf bug; skip in_type {in_type}')
            return
    else:
        data = np.zeros((2,), in_type)
        data[0] = this_min + 0j
        data[1] = this_max + 0j
    str_io = BytesIO()
    try:
        scale, inter, mn, mx = _calculate_scale(data, out_type, True)
    except ValueError as e:
        if DEBUG:
            warnings.warn(str((in_type, out_type, e)))
        return
    array_to_file(data, str_io, out_type, 0, inter, scale, mn, mx)
    data_back = array_from_file(data.shape, out_type, str_io)
    data_back = apply_read_scaling(data_back, scale, inter)
    assert np.allclose(big_floater(data), big_floater(data_back))
    scale32 = np.float32(scale)
    inter32 = np.float32(inter)
    if scale32 == np.inf or inter32 == np.inf:
        return
    data_back = array_from_file(data.shape, out_type, str_io)
    data_back = apply_read_scaling(data_back, scale32, inter32)
    info = type_info(in_type)
    out_min, out_max = (info['min'], info['max'])
    assert np.allclose(big_floater(data), big_floater(np.clip(data_back, out_min, out_max)))