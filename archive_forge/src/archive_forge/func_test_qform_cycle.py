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
def test_qform_cycle(self):
    img_klass = self.image_class
    img = img_klass(np.zeros((2, 3, 4)), None)
    hdr_back = self._qform_rt(img).header
    assert hdr_back['qform_code'] == 3
    assert hdr_back['sform_code'] == 4
    img = img_klass(np.zeros((2, 3, 4)), np.eye(4))
    hdr_back = self._qform_rt(img).header
    assert hdr_back['qform_code'] == 3
    assert hdr_back['sform_code'] == 4
    img.affine[0, 0] = 9
    img.to_file_map()
    img_back = img.from_file_map(img.file_map)
    exp_aff = np.diag([9, 1, 1, 1])
    assert_array_equal(img_back.affine, exp_aff)
    hdr_back = img.header
    assert_array_equal(hdr_back.get_sform(), exp_aff)
    assert_array_equal(hdr_back.get_qform(), exp_aff)