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
def test_reoriented_dim_info(self):
    arr = np.arange(24, dtype='f4').reshape((2, 3, 4))
    aff = np.diag([2, 3, 4, 1])
    simg = self.single_class(arr, aff)
    for freq, phas, slic in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 0, 1), (None, None, None), (0, 2, None), (0, None, None), (None, 2, 1), (None, None, 1)):
        simg.header.set_dim_info(freq, phas, slic)
        fdir = 'RAS'[freq] if freq is not None else None
        pdir = 'RAS'[phas] if phas is not None else None
        sdir = 'RAS'[slic] if slic is not None else None
        for ornt in ALL_ORNTS:
            rimg = simg.as_reoriented(np.array(ornt))
            axcode = aff2axcodes(rimg.affine)
            dirs = ''.join(axcode).replace('P', 'A').replace('I', 'S').replace('L', 'R')
            new_freq, new_phas, new_slic = rimg.header.get_dim_info()
            new_fdir = dirs[new_freq] if new_freq is not None else None
            new_pdir = dirs[new_phas] if new_phas is not None else None
            new_sdir = dirs[new_slic] if new_slic is not None else None
            assert (new_fdir, new_pdir, new_sdir) == (fdir, pdir, sdir)