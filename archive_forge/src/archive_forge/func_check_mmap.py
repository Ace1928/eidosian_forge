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
def check_mmap(hdr, offset, proxy_class, has_scaling=False, unscaled_is_view=True):
    """Assert that array proxies return memory maps as expected

    Parameters
    ----------
    hdr : object
        Image header instance
    offset : int
        Offset in bytes of image data in file (that we will write)
    proxy_class : class
        Class of image array proxy to test
    has_scaling : {False, True}
        True if the `hdr` says to apply scaling to the output data, False
        otherwise.
    unscaled_is_view : {True, False}
        True if getting the unscaled data returns a view of the array.  If
        False, then type of returned array will depend on whether numpy has the
        old viral (< 1.12) memmap behavior (returns memmap) or the new behavior
        (returns ndarray).  See: https://github.com/numpy/numpy/pull/7406
    """
    shape = hdr.get_data_shape()
    arr = np.arange(np.prod(shape), dtype=hdr.get_data_dtype()).reshape(shape)
    fname = 'test.bin'
    unscaled_really_mmap = unscaled_is_view
    scaled_really_mmap = unscaled_really_mmap and (not has_scaling)
    viral_memmap = memmap_after_ufunc()
    with InTemporaryDirectory():
        with open(fname, 'wb') as fobj:
            fobj.write(b' ' * offset)
            fobj.write(arr.tobytes(order='F'))
        for mmap, expected_mode in ((None, 'c'), (True, 'c'), ('c', 'c'), ('r', 'r'), (False, None)):
            kwargs = {}
            if mmap is not None:
                kwargs['mmap'] = mmap
            prox = proxy_class(fname, hdr, **kwargs)
            unscaled = prox.get_unscaled()
            back_data = np.asanyarray(prox)
            unscaled_is_mmap = isinstance(unscaled, np.memmap)
            back_is_mmap = isinstance(back_data, np.memmap)
            if expected_mode is None:
                assert not unscaled_is_mmap
                assert not back_is_mmap
            else:
                assert unscaled_is_mmap == (viral_memmap or unscaled_really_mmap)
                assert back_is_mmap == (viral_memmap or scaled_really_mmap)
                if scaled_really_mmap:
                    assert back_data.mode == expected_mode
            del prox, back_data
            with pytest.raises(TypeError):
                proxy_class(fname, hdr, True)
            with pytest.raises(ValueError):
                proxy_class(fname, hdr, mmap='rw')
            with pytest.raises(ValueError):
                proxy_class(fname, hdr, mmap='r+')