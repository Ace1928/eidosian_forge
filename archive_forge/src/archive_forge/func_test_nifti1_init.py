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
def test_nifti1_init():
    bio = BytesIO()
    shape = (2, 3, 4)
    hdr = Nifti1Header()
    arr = np.arange(24, dtype=np.int16).reshape(shape)
    write_raw_data(arr, hdr, bio)
    hdr.set_slope_inter(2, 10)
    ap = ArrayProxy(bio, hdr)
    assert ap.file_like == bio
    assert ap.shape == shape
    assert_array_equal(np.asarray(ap), arr * 2.0 + 10)
    with InTemporaryDirectory():
        f = open('test.nii', 'wb')
        write_raw_data(arr, hdr, f)
        f.close()
        ap = ArrayProxy('test.nii', hdr)
        assert ap.file_like == 'test.nii'
        assert ap.shape == shape
        assert_array_equal(np.asarray(ap), arr * 2.0 + 10)