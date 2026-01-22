import contextlib
import gzip
import pickle
from io import BytesIO
from unittest import mock
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from packaging.version import Version
from .. import __version__
from ..arrayproxy import ArrayProxy, get_obj_dtype, is_proxy, reshape_dataobj
from ..deprecator import ExpiredDeprecationError
from ..nifti1 import Nifti1Header, Nifti1Image
from ..openers import ImageOpener
from ..testing import memmap_after_ufunc
from ..tmpdirs import InTemporaryDirectory
from .test_fileslice import slicer_samples
from .test_openers import patch_indexed_gzip
def test_tuplespec():
    bio = BytesIO()
    shape = [2, 3, 4]
    dtype = np.int32
    arr = np.arange(24, dtype=dtype).reshape(shape)
    bio.seek(16)
    bio.write(arr.tobytes(order='F'))
    hdr = FunkyHeader(shape)
    tuple_spec = (hdr.get_data_shape(), hdr.get_data_dtype(), hdr.get_data_offset(), 1.0, 0.0)
    ap_header = ArrayProxy(bio, hdr)
    ap_tuple = ArrayProxy(bio, tuple_spec)
    for prop in ('shape', 'dtype', 'offset', 'slope', 'inter', 'is_proxy'):
        assert getattr(ap_header, prop) == getattr(ap_tuple, prop)
    for method, args in (('get_unscaled', ()), ('__array__', ()), ('__getitem__', ((0, 2, 1),))):
        assert_array_equal(getattr(ap_header, method)(*args), getattr(ap_tuple, method)(*args))
    for n in range(2, 5):
        ArrayProxy(bio, tuple_spec[:n])
    with pytest.raises(TypeError):
        ArrayProxy(bio, ())
    with pytest.raises(TypeError):
        ArrayProxy(bio, tuple_spec[:1])
    with pytest.raises(TypeError):
        ArrayProxy(bio, tuple_spec + ('error',))