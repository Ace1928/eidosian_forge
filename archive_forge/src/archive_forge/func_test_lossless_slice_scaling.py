import os
import unittest
from unittest import mock
import numpy as np
import pytest
import nibabel as nb
from nibabel.cmdline.roi import lossless_slice, main, parse_slice
from nibabel.testing import data_path
def test_lossless_slice_scaling(tmp_path):
    fname = tmp_path / 'image.nii'
    img = nb.Nifti1Image(np.random.uniform(-20000, 20000, (5, 5, 5, 5)), affine=np.eye(4))
    img.header.set_data_dtype('int16')
    img.to_filename(fname)
    img1 = nb.load(fname)
    sliced_fname = tmp_path / 'sliced.nii'
    lossless_slice(img1, (slice(None), slice(None), slice(2, 4))).to_filename(sliced_fname)
    img2 = nb.load(sliced_fname)
    assert np.array_equal(img1.get_fdata()[:, :, 2:4], img2.get_fdata())
    assert np.array_equal(img1.dataobj.get_unscaled()[:, :, 2:4], img2.dataobj.get_unscaled())
    assert img1.dataobj.slope == img2.dataobj.slope
    assert img1.dataobj.inter == img2.dataobj.inter