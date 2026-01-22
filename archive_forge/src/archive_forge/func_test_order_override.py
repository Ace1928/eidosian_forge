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
@pytest.mark.parametrize('order', ('C', 'F'))
def test_order_override(order):
    shape = (15, 16, 17)
    arr = np.arange(np.prod(shape)).reshape(shape)
    fobj = BytesIO()
    fobj.write(arr.tobytes(order=order))
    for klass in (ArrayProxy, CArrayProxy):
        prox = klass(fobj, (shape, arr.dtype), order=order)
        assert prox.order == order
        sliceobj = (None, slice(None), 1, -1)
        assert_array_equal(arr[sliceobj], prox[sliceobj])