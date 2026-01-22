import itertools
import unittest
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..optpkg import optional_package
from ..casting import sctypes_aliases, shared_range, type_info
from ..spatialimages import HeaderDataError
from ..spm99analyze import HeaderTypeError, Spm99AnalyzeHeader, Spm99AnalyzeImage
from ..testing import (
from ..volumeutils import _dt_min_max, apply_read_scaling
from . import test_analyze
@pytest.mark.parametrize('in_dtype', FLOAT_TYPES + IUINT_TYPES)
def test_no_scaling(self, in_dtype, supported_dtype):
    img_class = self.image_class
    hdr_class = img_class.header_class
    hdr = hdr_class()
    slope = 2
    inter = 10 if hdr.has_data_intercept else 0
    mn_in, mx_in = _dt_min_max(in_dtype)
    mn = -1 if np.dtype(in_dtype).kind != 'u' else 0
    arr = np.array([mn_in, mn, 0, 1, 10, mx_in], dtype=in_dtype)
    img = img_class(arr, np.eye(4), hdr)
    img.set_data_dtype(supported_dtype)
    img.header.set_slope_inter(slope, inter)
    with np.errstate(invalid='ignore'):
        rt_img = bytesio_round_trip(img)
    with suppress_warnings():
        back_arr = np.asanyarray(rt_img.dataobj)
    exp_back = arr.copy()
    if supported_dtype in IUINT_TYPES:
        if in_dtype in FLOAT_TYPES:
            exp_back = exp_back.astype(float)
            with np.errstate(invalid='ignore'):
                exp_back = np.round(exp_back)
            if in_dtype in FLOAT_TYPES:
                exp_back = np.clip(exp_back, *shared_range(float, supported_dtype))
        else:
            mn_out, mx_out = _dt_min_max(supported_dtype)
            if (mn_in, mx_in) != (mn_out, mx_out):
                exp_back = np.clip(exp_back, max(mn_in, mn_out), min(mx_in, mx_out))
    if supported_dtype in COMPLEX_TYPES:
        exp_back = exp_back.astype(supported_dtype)
    else:
        exp_back = exp_back.astype(float)
    with suppress_warnings():
        assert_allclose_safely(back_arr, exp_back * slope + inter)