import io
import pathlib
import sys
import warnings
from functools import partial
from itertools import product
import numpy as np
from ..optpkg import optional_package
import unittest
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal
from nibabel.arraywriters import WriterError
from nibabel.testing import (
from .. import (
from ..casting import sctypes
from ..spatialimages import SpatialImage
from ..tmpdirs import InTemporaryDirectory
from .test_api_validators import ValidateAPI
from .test_brikhead import EXAMPLE_IMAGES as AFNI_EXAMPLE_IMAGES
from .test_minc1 import EXAMPLE_IMAGES as MINC1_EXAMPLE_IMAGES
from .test_minc2 import EXAMPLE_IMAGES as MINC2_EXAMPLE_IMAGES
from .test_parrec import EXAMPLE_IMAGES as PARREC_EXAMPLE_IMAGES
def validate_filenames(self, imaker, params):
    if not self.can_save:
        raise unittest.SkipTest
    img = imaker()
    img.set_data_dtype(np.float32)
    img.file_map = None
    rt_img = bytesio_round_trip(img)
    assert_array_equal(img.shape, rt_img.shape)
    assert_almost_equal(img.get_fdata(), rt_img.get_fdata())
    assert_almost_equal(np.asanyarray(img.dataobj), np.asanyarray(rt_img.dataobj))
    klass = type(img)
    rt_img.file_map = bytesio_filemap(klass)
    rt_img.to_file_map()
    rt_rt_img = klass.from_file_map(rt_img.file_map)
    assert_almost_equal(img.get_fdata(), rt_rt_img.get_fdata())
    assert_almost_equal(np.asanyarray(img.dataobj), np.asanyarray(rt_img.dataobj))
    fname = 'an_image' + self.standard_extension
    for path in (fname, pathlib.Path(fname)):
        img.set_filename(path)
        assert img.get_filename() == str(path)
        assert img.file_map['image'].filename == str(path)
    fname = 'another_image' + self.standard_extension
    for path in (fname, pathlib.Path(fname)):
        with InTemporaryDirectory():
            with clear_and_catch_warnings():
                warnings.filterwarnings('error', category=DeprecationWarning, module='nibabel.*')
                img.to_filename(path)
                rt_img = img.__class__.from_filename(path)
            assert_array_equal(img.shape, rt_img.shape)
            assert_almost_equal(img.get_fdata(), rt_img.get_fdata())
            assert_almost_equal(np.asanyarray(img.dataobj), np.asanyarray(rt_img.dataobj))
            del rt_img