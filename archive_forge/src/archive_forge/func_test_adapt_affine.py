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
def test_adapt_affine():
    aff_3d = from_matvec(np.arange(9).reshape((3, 3)), [11, 12, 13])
    assert_array_equal(adapt_affine(aff_3d, 3), aff_3d)
    assert_array_equal(adapt_affine(aff_3d, 4), [[0, 1, 2, 0, 11], [3, 4, 5, 0, 12], [6, 7, 8, 0, 13], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    aff_4d = from_matvec(np.arange(16).reshape((4, 4)), [11, 12, 13, 14])
    assert_array_equal(adapt_affine(aff_4d, 4), aff_4d)
    assert_array_equal(adapt_affine(aff_3d, 2), [[0, 1, 11], [3, 4, 12], [6, 7, 13], [0, 0, 1]])
    assert_array_equal(adapt_affine(aff_3d, 1), [[0, 11], [3, 12], [6, 13], [0, 1]])
    aff_2d = from_matvec(np.arange(4).reshape((2, 2)), [11, 12])
    assert_array_equal(adapt_affine(aff_2d, 2), aff_2d)