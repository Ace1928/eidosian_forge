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
def test_loadsave_cycle(self):
    nim = self.module.load(self.example_file)
    hdr = nim.header
    exts_container = hdr.extensions
    assert len(exts_container) > 0
    lnim = bytesio_round_trip(nim)
    hdr = lnim.header
    lexts_container = hdr.extensions
    assert exts_container == lexts_container
    data = np.ones((2, 3, 4, 5), dtype='int16')
    img = self.single_class(data, np.eye(4))
    hdr = img.header
    assert hdr.get_data_dtype() == np.int16
    assert_array_equal(hdr.get_slope_inter(), (None, None))
    hdr.set_slope_inter(2, 8)
    assert hdr.get_slope_inter() == (2, 8)
    wnim = self.single_class(data, np.eye(4), header=hdr)
    assert wnim.get_data_dtype() == np.int16
    assert wnim.header.get_slope_inter() == (None, None)
    wnim.header.set_slope_inter(2, 8)
    assert wnim.header.get_slope_inter() == (2, 8)
    lnim = bytesio_round_trip(wnim)
    assert lnim.get_data_dtype() == np.int16
    assert_array_equal(lnim.get_fdata(), data * 2.0 + 8.0)
    assert lnim.header.get_slope_inter() == (None, None)
    assert (lnim.dataobj.slope, lnim.dataobj.inter) == (2, 8)