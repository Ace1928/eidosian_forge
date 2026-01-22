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
def validate_mmap_parameter(self, imaker, params):
    img = imaker()
    fname = img.get_filename()
    with InTemporaryDirectory():
        if fname is None:
            if not img.rw or not img.valid_exts:
                return
            fname = 'image' + img.valid_exts[0]
            img.to_filename(fname)
        rt_img = img.__class__.from_filename(fname, mmap=True)
        assert_almost_equal(img.get_fdata(), rt_img.get_fdata())
        rt_img = img.__class__.from_filename(fname, mmap=False)
        assert_almost_equal(img.get_fdata(), rt_img.get_fdata())
        rt_img = img.__class__.from_filename(fname, mmap='c')
        assert_almost_equal(img.get_fdata(), rt_img.get_fdata())
        rt_img = img.__class__.from_filename(fname, mmap='r')
        assert_almost_equal(img.get_fdata(), rt_img.get_fdata())
        with pytest.raises(ValueError):
            img.__class__.from_filename(fname, mmap='r+')
        with pytest.raises(ValueError):
            img.__class__.from_filename(fname, mmap='invalid')
        del rt_img