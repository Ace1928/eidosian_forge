import logging
import pathlib
import shutil
from io import BytesIO
from os.path import dirname
from os.path import join as pjoin
from tempfile import mkdtemp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import (
from .. import analyze as ana
from .. import loadsave as nils
from .. import nifti1 as ni1
from .. import spm2analyze as spm2
from .. import spm99analyze as spm99
from ..optpkg import optional_package
from ..spatialimages import SpatialImage
from ..testing import deprecated_to, expires
from ..tmpdirs import InTemporaryDirectory
from ..volumeutils import native_code, swapped_code
def test_negative_load_save():
    shape = (1, 2, 5)
    data = np.arange(10).reshape(shape) - 10.0
    affine = np.eye(4)
    hdr = ni1.Nifti1Header()
    hdr.set_data_dtype(np.int16)
    img = Nifti1Image(data, affine, hdr)
    str_io = BytesIO()
    img.file_map['image'].fileobj = str_io
    img.to_file_map()
    str_io.seek(0)
    re_img = Nifti1Image.from_file_map(img.file_map)
    assert_array_almost_equal(re_img.get_fdata(), data, 4)