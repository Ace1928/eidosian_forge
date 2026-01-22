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
def test_rt_bias(self):
    rng = np.random.RandomState(20111214)
    mu, std, count = (100, 10, 100)
    arr = rng.normal(mu, std, size=(count,))
    eps = np.finfo(np.float32).eps
    aff = np.eye(4)
    for in_dt in (np.float32, np.float64):
        arr_t = arr.astype(in_dt)
        for out_dt in IUINT_TYPES:
            img = self.single_class(arr_t, aff)
            img_back = bytesio_round_trip(img)
            arr_back_sc = img_back.get_fdata()
            slope, inter = img_back.header.get_slope_inter()
            bias = np.mean(arr_t - arr_back_sc)
            max_miss = rt_err_estimate(arr_t, arr_back_sc.dtype, slope, inter)
            bias_thresh = np.max([max_miss / np.sqrt(count), eps])
            assert np.abs(bias) < bias_thresh