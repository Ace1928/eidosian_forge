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
def test_big_scaling(self):
    hdr = self.header_class()
    hdr.set_data_shape((1, 1, 1))
    hdr.set_data_dtype(np.int16)
    sio = BytesIO()
    dtt = np.float32
    data = np.array([type_info(dtt)['max']], dtype=dtt)[:, None, None]
    hdr.data_to_fileobj(data, sio)
    data_back = hdr.data_from_fileobj(sio)
    assert np.allclose(data, data_back)