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
def test_qform_sform(self):
    HC = self.header_class
    hdr = HC()
    assert_array_equal(hdr.get_qform(), np.eye(4))
    empty_sform = np.zeros((4, 4))
    empty_sform[-1, -1] = 1
    assert_array_equal(hdr.get_sform(), empty_sform)
    assert hdr.get_qform(coded=True) == (None, 0)
    assert hdr.get_sform(coded=True) == (None, 0)
    nice_aff = np.diag([2, 3, 4, 1])
    another_aff = np.diag([3, 4, 5, 1])
    nasty_aff = from_matvec(np.arange(9).reshape((3, 3)), [9, 10, 11])
    nasty_aff[0, 0] = 1
    fixed_aff = unshear_44(nasty_aff)
    assert not np.allclose(fixed_aff, nasty_aff)
    for in_meth, out_meth in ((hdr.set_qform, hdr.get_qform), (hdr.set_sform, hdr.get_sform)):
        in_meth(nice_aff, 2)
        aff, code = out_meth(coded=True)
        assert_array_equal(aff, nice_aff)
        assert code == 2
        assert_array_equal(out_meth(), nice_aff)
        in_meth(another_aff, 0)
        assert out_meth(coded=True) == (None, 0)
        assert_array_almost_equal(out_meth(), another_aff)
        in_meth(nice_aff)
        aff, code = out_meth(coded=True)
        assert code == 2
        in_meth(nice_aff, 1)
        in_meth(nice_aff)
        aff, code = out_meth(coded=True)
        assert code == 1
        assert_array_equal(aff, nice_aff)
        in_meth(None, 3)
        aff, code = out_meth(coded=True)
        assert_array_equal(aff, nice_aff)
        assert code == 3
        in_meth(None, 0)
        assert out_meth(coded=True) == (None, 0)
        in_meth(None)
        assert out_meth(coded=True) == (None, 0)
        in_meth(nice_aff.tolist())
        assert_array_equal(out_meth(), nice_aff)
    hdr.set_qform(nasty_aff, 1)
    assert_array_almost_equal(hdr.get_qform(), fixed_aff)
    with pytest.raises(HeaderDataError):
        hdr.set_qform(nasty_aff, 1, False)
    hdr.set_sform(None)
    hdr.set_qform(nice_aff, 1)
    assert hdr.get_sform(coded=True) == (None, 0)
    hdr.set_sform(nasty_aff, 1)
    aff, code = hdr.get_sform(coded=True)
    assert_array_equal(aff, nasty_aff)
    assert code == 1