import unittest
from glob import glob
from os.path import basename, exists
from os.path import join as pjoin
from os.path import splitext
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from .. import load as top_load
from ..affines import voxel_sizes
from ..parrec import load
from .nibabel_data import get_nibabel_data, needs_nibabel_data
@needs_nibabel_data('parrec_oblique')
def test_oblique_loading():
    for par in glob(pjoin(OBLIQUE, 'PARREC', '*.PAR')):
        par_root, ext = splitext(basename(par))
        pimg = load(par)
        assert pimg.shape == (560, 560, 1)
        nifti_fname = pjoin(OBLIQUE, 'NIFTI', par_root + '.nii')
        nimg = top_load(nifti_fname)
        assert_almost_equal(nimg.affine[:3, :3], pimg.affine[:3, :3], 3)
        aff_off = pimg.affine[:3, 3] - nimg.affine[:3, 3]
        vox_sizes = voxel_sizes(nimg.affine)
        assert np.all(np.abs(aff_off / vox_sizes) <= 0.5)
        assert np.allclose(pimg.dataobj, nimg.dataobj)