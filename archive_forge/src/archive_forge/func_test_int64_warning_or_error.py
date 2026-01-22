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
def test_int64_warning_or_error(self):
    img_klass = self.image_class
    hdr_klass = img_klass.header_class
    for dtype in (np.int64, np.uint64):
        data = np.arange(24, dtype=dtype).reshape((2, 3, 4))
        if cmp_pkg_version('5.0') <= 0:
            cm = pytest.raises(ValueError)
        else:
            cm = pytest.warns(FutureWarning)
        with cm:
            img_klass(data, np.eye(4))
        with clear_and_catch_warnings():
            warnings.simplefilter('error')
            img_klass(data, np.eye(4), dtype=dtype)
            hdr = hdr_klass()
            hdr.set_data_dtype(dtype)
            img_klass(data, np.eye(4), hdr)