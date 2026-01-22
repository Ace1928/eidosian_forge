import itertools
import logging
import os
import pickle
import re
from io import BytesIO, StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import imageglobals
from ..analyze import AnalyzeHeader, AnalyzeImage
from ..arraywriters import WriterError
from ..casting import sctypes_aliases
from ..nifti1 import Nifti1Header
from ..optpkg import optional_package
from ..spatialimages import HeaderDataError, HeaderTypeError, supported_np_types
from ..testing import (
from ..tmpdirs import InTemporaryDirectory
from . import test_spatialimages as tsi
from . import test_wrapstruct as tws
def test_read_write_data(self):
    hdr = self.header_class()
    bytes = hdr.data_from_fileobj(BytesIO())
    assert len(bytes) == 0
    str_io = BytesIO()
    hdr.data_to_fileobj([], str_io)
    assert str_io.getvalue() == b''
    with pytest.raises(HeaderDataError):
        hdr.data_to_fileobj(np.zeros(3), str_io)
    hdr.set_data_shape((1, 2, 3))
    hdr.set_data_dtype(np.float32)
    S = BytesIO()
    data = np.arange(6, dtype=np.float64)
    with pytest.raises(HeaderDataError):
        hdr.data_to_fileobj(data, S)
    data = data.reshape((1, 2, 3))
    with pytest.raises(HeaderDataError):
        hdr.data_to_fileobj(data[:, :, :-1], S)
    with pytest.raises(HeaderDataError):
        hdr.data_to_fileobj(data[:, :-1, :], S)
    hdr.data_to_fileobj(data, S)
    data_back = hdr.data_from_fileobj(S)
    assert_array_almost_equal(data, data_back)
    assert hdr.get_data_dtype() == data_back.dtype
    S2 = BytesIO()
    hdr2 = hdr.as_byteswapped()
    hdr2.set_data_dtype(np.float32)
    hdr2.set_data_shape((1, 2, 3))
    hdr2.data_to_fileobj(data, S2)
    data_back2 = hdr2.data_from_fileobj(S2)
    assert_array_almost_equal(data_back, data_back2)
    assert data_back.dtype.name == data_back2.dtype.name
    assert data.dtype.byteorder != data_back2.dtype.byteorder
    hdr.set_data_dtype(np.uint8)
    S3 = BytesIO()
    with np.errstate(invalid='ignore'):
        hdr.data_to_fileobj(data, S3, rescale=False)
    data_back = hdr.data_from_fileobj(S3)
    assert_array_almost_equal(data, data_back)
    if not hdr.has_data_slope:
        with pytest.raises(HeaderTypeError):
            hdr.data_to_fileobj(data, S3)
        with pytest.raises(HeaderTypeError):
            hdr.data_to_fileobj(data, S3, rescale=True)
    data = np.arange(6, dtype=np.float64).reshape((1, 2, 3)) + 0.5
    with np.errstate(invalid='ignore'):
        hdr.data_to_fileobj(data, S3, rescale=False)
    data_back = hdr.data_from_fileobj(S3)
    assert not np.allclose(data, data_back)
    dtype = np.dtype([('R', 'uint8'), ('G', 'uint8'), ('B', 'uint8')])
    data = np.ones((1, 2, 3), dtype)
    hdr.set_data_dtype(dtype)
    S4 = BytesIO()
    hdr.data_to_fileobj(data, S4)
    data_back = hdr.data_from_fileobj(S4)
    assert_array_equal(data, data_back)