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
def test_sform(self):
    ehdr = self.header_class()
    ehdr.set_sform(A)
    sA = ehdr.get_sform()
    assert np.allclose(A, sA, atol=1e-05)
    xfas = nifti1.xform_codes
    assert ehdr['sform_code'] == xfas['aligned']
    ehdr.set_sform(A, 'scanner')
    assert ehdr['sform_code'] == xfas['scanner']
    ehdr.set_sform(A, xfas['aligned'])
    assert ehdr['sform_code'] == xfas['aligned']