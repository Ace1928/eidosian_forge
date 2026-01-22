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
def test_set_qform(self):
    img = self.image_class(np.zeros((2, 3, 4)), np.diag([2.2, 3.3, 4.3, 1]))
    hdr = img.header
    new_affine = np.diag([1.1, 1.1, 1.1, 1])
    assert_array_almost_equal(img.affine, hdr.get_best_affine())
    aff_affine = np.diag([3.3, 4.5, 6.6, 1])
    img.affine[:] = aff_affine
    assert_array_almost_equal(img.affine, aff_affine)
    img.set_qform(new_affine, 1)
    assert_array_almost_equal(img.get_qform(), new_affine)
    assert hdr['qform_code'] == 1
    assert_array_almost_equal(img.get_qform(), new_affine)
    qaff, code = img.get_qform(coded=True)
    assert code == 1
    assert_array_almost_equal(qaff, new_affine)
    assert_array_almost_equal(img.affine, hdr.get_best_affine())
    img.affine[:] = aff_affine
    img.set_qform(new_affine, 1, update_affine=False)
    assert_array_almost_equal(img.affine, aff_affine)
    assert_array_almost_equal(hdr.get_zooms(), [1.1, 1.1, 1.1])
    img.set_qform(None)
    qaff, code = img.get_qform(coded=True)
    assert (qaff, code) == (None, 0)
    assert_array_almost_equal(hdr.get_zooms(), [1.1, 1.1, 1.1])
    assert_array_almost_equal(img.affine, hdr.get_best_affine())
    img.set_sform(None)
    img.set_qform(new_affine, 1)
    qaff, code = img.get_qform(coded=True)
    assert code == 1
    assert_array_almost_equal(img.affine, new_affine)
    new_affine[0, 1] = 2
    img.set_qform(new_affine, 2)
    with pytest.raises(HeaderDataError):
        img.set_qform(new_affine, 2, False)
    with pytest.raises(TypeError):
        img.get_qform(strange=True)
    img = self.image_class(np.zeros((2, 3, 4)), None)
    new_affine = np.eye(4)
    img.set_qform(new_affine, 2)
    assert_array_almost_equal(img.affine, img.header.get_best_affine())
    img.set_sform(None, update_affine=True)
    assert_array_almost_equal(img.affine, new_affine)