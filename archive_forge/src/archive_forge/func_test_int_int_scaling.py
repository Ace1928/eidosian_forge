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
def test_int_int_scaling(self):
    img_class = self.image_class
    arr = np.array([-1, 0, 256], dtype=np.int16)[:, None, None]
    img = img_class(arr, np.eye(4))
    hdr = img.header
    img.set_data_dtype(np.uint8)
    self._set_raw_scaling(hdr, 1, 0 if hdr.has_data_intercept else None)
    img_rt = bytesio_round_trip(img)
    assert_array_equal(img_rt.get_fdata(), np.clip(arr, 0, 255))