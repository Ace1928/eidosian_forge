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
def test_get_obj_dtype():
    bio = BytesIO()
    shape = (2, 3, 4)
    hdr = Nifti1Header()
    arr = np.arange(24, dtype=np.int16).reshape(shape)
    write_raw_data(arr, hdr, bio)
    hdr.set_slope_inter(2, 10)
    prox = ArrayProxy(bio, hdr)
    assert get_obj_dtype(prox) == np.dtype('float64')
    assert get_obj_dtype(np.array(prox)) == np.dtype('float64')
    hdr.set_slope_inter(1, 0)
    prox = ArrayProxy(bio, hdr)
    assert get_obj_dtype(prox) == np.dtype('int16')
    assert get_obj_dtype(np.array(prox)) == np.dtype('int16')

    class ArrGiver:

        def __array__(self):
            return arr
    assert get_obj_dtype(ArrGiver()) == np.dtype('int16')