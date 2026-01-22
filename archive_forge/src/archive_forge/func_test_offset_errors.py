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
def test_offset_errors(self):
    IC = self.image_class
    arr = np.arange(24, dtype='f4').reshape((2, 3, 4))
    img = IC(arr, np.eye(4))
    assert img.header.get_data_offset() == 0
    img_rt = bytesio_round_trip(img)
    assert img_rt.header.get_data_offset() == 0
    fm = bytesio_filemap(IC)
    img.header.set_data_offset(16)
    with pytest.raises(HeaderDataError):
        img.to_file_map(fm)