import logging
from os.path import dirname
from os.path import join as pjoin
import numpy as np
import numpy.linalg as npl
from nibabel.optpkg import optional_package
import unittest
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
import nibabel as nib
from nibabel.affines import AffineError, apply_affine, from_matvec, to_matvec, voxel_sizes
from nibabel.eulerangles import euler2mat
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image
from nibabel.orientations import aff2axcodes, inv_ornt_aff
from nibabel.processing import (
from nibabel.testing import assert_allclose_safely
from nibabel.tests.test_spaces import assert_all_in, get_outspace_params
from .test_imageclasses import MINC_3DS, MINC_4DS
def test_sigma2fwhm():
    assert_almost_equal(sigma2fwhm(1), 2.35482)
    assert_almost_equal(sigma2fwhm([1, 2, 3]), np.arange(1, 4) * 2.35482)
    assert_almost_equal(fwhm2sigma(2.35482), 1)
    assert_almost_equal(fwhm2sigma(np.arange(1, 4) * 2.35482), [1, 2, 3])
    fwhm = np.arange(1.0, 5.0, 0.1)
    sigma = np.arange(1.0, 5.0, 0.1)
    assert np.allclose(sigma2fwhm(fwhm2sigma(fwhm)), fwhm)
    assert np.allclose(fwhm2sigma(sigma2fwhm(sigma)), sigma)