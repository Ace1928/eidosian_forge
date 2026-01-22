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
def validate_data_interface(self, imaker, params):
    img = imaker()
    assert img.shape == img.dataobj.shape
    assert img.ndim == len(img.shape)
    assert_data_similar(img.dataobj, params)
    for meth_name in self.meth_names:
        if params['is_proxy']:
            self._check_proxy_interface(imaker, meth_name)
        else:
            self._check_array_interface(imaker, meth_name)
        method = getattr(img, meth_name)
        assert img.shape == method().shape
        assert img.ndim == method().ndim
        with pytest.raises(ValueError):
            method(caching='something')
    fake_data = np.zeros(img.shape, dtype=img.get_data_dtype())
    with pytest.raises(AttributeError):
        img.dataobj = fake_data
    with pytest.raises(AttributeError):
        img.in_memory = False