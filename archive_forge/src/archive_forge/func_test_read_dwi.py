from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nibabel.optpkg import optional_package
from .. import dicomreaders as didr
from .test_dicomwrappers import DATA, EXPECTED_AFFINE, EXPECTED_PARAMS, IO_DATA_PATH
def test_read_dwi():
    img = didr.mosaic_to_nii(DATA)
    arr = img.get_fdata()
    assert arr.shape == (128, 128, 48)
    assert_array_almost_equal(img.affine, EXPECTED_AFFINE)