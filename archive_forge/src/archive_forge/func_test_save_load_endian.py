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
def test_save_load_endian():
    shape = (2, 4, 6)
    affine = np.diag([1, 2, 3, 1])
    data = np.arange(np.prod(shape), dtype='f4').reshape(shape)
    img = Nifti1Image(data, affine)
    assert img.header.endianness == native_code
    img2 = round_trip(img)
    assert img2.header.endianness == native_code
    assert_array_equal(img2.get_fdata(), data)
    assert_array_equal(np.asanyarray(img2.dataobj), data)
    bs_hdr = img.header.as_byteswapped()
    bs_img = Nifti1Image(data, affine, bs_hdr)
    assert bs_img.header.endianness == swapped_code
    assert_array_equal(bs_img.get_fdata(), data)
    assert_array_equal(np.asanyarray(bs_img.dataobj), data)
    cbs_img = AnalyzeImage.from_image(bs_img)
    cbs_hdr = cbs_img.header
    assert cbs_hdr.endianness == native_code
    cbs_img2 = Nifti1Image.from_image(cbs_img)
    cbs_hdr2 = cbs_img2.header
    assert cbs_hdr2.endianness == native_code
    bs_img2 = round_trip(bs_img)
    bs_data2 = np.asanyarray(bs_img2.dataobj)
    bs_fdata2 = bs_img2.get_fdata()
    assert bs_data2.dtype.byteorder == swapped_code
    assert bs_img2.header.endianness == swapped_code
    assert_array_equal(bs_data2, data)
    assert bs_fdata2.dtype.byteorder != swapped_code
    assert_array_equal(bs_fdata2, data)
    mixed_img = Nifti1Image(bs_data2, affine)
    assert mixed_img.header.endianness == native_code
    m_img2 = round_trip(mixed_img)
    assert m_img2.header.endianness == native_code
    assert_array_equal(m_img2.get_fdata(), data)