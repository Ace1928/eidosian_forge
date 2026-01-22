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
@pytest.mark.xfail(sys.version_info >= (3, 12), reason='Response type for file: urls is not a stream in Python 3.12')
def validate_from_file_url(self, imaker, params):
    tmp_path = self.tmp_path
    img = imaker()
    import uuid
    fname = tmp_path / f'img-{uuid.uuid4()}{self.standard_extension}'
    img.to_filename(fname)
    rt_img = img.__class__.from_url(f'file:///{fname}')
    assert self._header_eq(img.header, rt_img.header)
    assert np.array_equal(img.get_fdata(), rt_img.get_fdata())
    del img
    del rt_img