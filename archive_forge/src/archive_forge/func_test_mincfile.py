import bz2
import gzip
import types
import warnings
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import Nifti1Image, load, minc1
from ..deprecated import ModuleProxy
from ..deprecator import ExpiredDeprecationError
from ..externals.netcdf import netcdf_file
from ..minc1 import Minc1File, Minc1Image, MincHeader
from ..optpkg import optional_package
from ..testing import assert_data_similar, clear_and_catch_warnings, data_path
from ..tmpdirs import InTemporaryDirectory
from . import test_spatialimages as tsi
from .test_fileslice import slicer_samples
def test_mincfile(self):
    for tp in self.test_files:
        mnc_obj = self.opener(tp['fname'], 'r')
        mnc = self.file_class(mnc_obj)
        assert mnc.get_data_dtype().type == tp['dtype']
        assert mnc.get_data_shape() == tp['shape']
        assert mnc.get_zooms() == tp['zooms']
        assert_array_equal(mnc.get_affine(), tp['affine'])
        data = mnc.get_scaled_data()
        assert data.shape == tp['shape']
        del mnc, data