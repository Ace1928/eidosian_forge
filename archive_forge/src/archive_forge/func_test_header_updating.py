import io
import os
import pathlib
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from ... import imageglobals
from ...fileholders import FileHolder
from ...openers import ImageOpener
from ...spatialimages import HeaderDataError
from ...testing import data_path
from ...tests import test_spatialimages as tsi
from ...tests import test_wrapstruct as tws
from ...tmpdirs import InTemporaryDirectory
from ...volumeutils import sys_is_le
from ...wrapstruct import WrapStructError
from .. import load, save
from ..mghformat import MGHError, MGHHeader, MGHImage
def test_header_updating():
    mgz = load(MGZ_FNAME)
    hdr = mgz.header
    exp_aff = np.loadtxt(io.BytesIO(b'\n    1.0000   2.0000   3.0000   -13.0000\n    2.0000   3.0000   1.0000   -11.5000\n    3.0000   1.0000   2.0000   -11.5000\n    0.0000   0.0000   0.0000     1.0000'))
    assert_almost_equal(mgz.affine, exp_aff, 6)
    assert_almost_equal(hdr.get_affine(), exp_aff, 6)
    assert np.all(hdr['delta'] == 1)
    assert_almost_equal(hdr['Mdc'].T, exp_aff[:3, :3])
    img_fobj = io.BytesIO()
    mgz2 = _mgh_rt(mgz, img_fobj)
    hdr2 = mgz2.header
    assert_almost_equal(hdr2.get_affine(), exp_aff, 6)
    assert_array_equal(hdr2['delta'], 1)
    exp_aff_d = exp_aff.copy()
    exp_aff_d[0, -1] = -14
    mgz2._affine[:] = exp_aff_d
    mgz2.update_header()
    assert_almost_equal(hdr2.get_affine(), exp_aff_d, 6)
    RZS = exp_aff_d[:3, :3]
    assert_almost_equal(hdr2['delta'], np.sqrt(np.sum(RZS ** 2, axis=0)))
    assert_almost_equal(hdr2['Mdc'].T, RZS / hdr2['delta'])