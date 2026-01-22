import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.metrics import structural_similarity
def test_ssim_warns_about_data_range():
    mssim = structural_similarity(cam, cam_noisy)
    with expected_warnings(['Setting data_range based on im1.dtype']):
        mssim_uint16 = structural_similarity(cam.astype(np.uint16), cam_noisy.astype(np.uint16))
        assert mssim_uint16 > 0.99
    with expected_warnings(['Setting data_range based on im1.dtype', 'Inputs have mismatched dtypes']):
        mssim_mixed = structural_similarity(cam, cam_noisy.astype(np.int32))
    mssim_mixed = structural_similarity(cam, cam_noisy.astype(np.float32), data_range=255)
    assert_almost_equal(mssim, mssim_mixed)