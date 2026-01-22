import os
import struct
import unittest
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from nibabel import nifti1 as nifti1
from nibabel.affines import from_matvec
from nibabel.casting import have_binary128, type_info
from nibabel.eulerangles import euler2mat
from nibabel.nifti1 import (
from nibabel.optpkg import optional_package
from nibabel.pkg_info import cmp_pkg_version
from nibabel.spatialimages import HeaderDataError
from nibabel.tmpdirs import InTemporaryDirectory
from ..freesurfer import load as mghload
from ..orientations import aff2axcodes
from ..testing import (
from . import test_analyze as tana
from . import test_spm99analyze as tspm
from .nibabel_data import get_nibabel_data, needs_nibabel_data
from .test_arraywriters import IUINT_TYPES, rt_err_estimate
from .test_orientations import ALL_ORNTS
def test_load_save(self):
    IC = self.image_class
    img_ext = IC.files_types[0][1]
    shape = (2, 4, 6)
    npt = np.float32
    data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
    affine = np.diag([1, 2, 3, 1])
    img = IC(data, affine)
    assert img.header.get_data_offset() == 0
    assert img.shape == shape
    img.set_data_dtype(npt)
    img2 = bytesio_round_trip(img)
    assert_array_equal(img2.get_fdata(), data)
    with InTemporaryDirectory() as tmpdir:
        for ext in ('', '.gz', '.bz2'):
            fname = os.path.join(tmpdir, 'test' + img_ext + ext)
            img.to_filename(fname)
            img3 = IC.load(fname)
            assert isinstance(img3, img.__class__)
            assert_array_equal(img3.get_fdata(), data)
            assert img3.header == img.header
            assert isinstance(np.asanyarray(img3.dataobj), np.memmap if ext == '' else np.ndarray)
            del img3