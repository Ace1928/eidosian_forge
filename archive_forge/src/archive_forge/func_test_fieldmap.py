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
@needs_nibabel_data('nitest-balls1')
def test_fieldmap():
    fieldmap_par = pjoin(BALLS, 'PARREC', 'fieldmap.PAR')
    fieldmap_nii = pjoin(BALLS, 'NIFTI', 'fieldmap.nii.gz')
    load(fieldmap_par)
    top_load(fieldmap_nii)
    raise unittest.SkipTest('Fieldmap remains puzzling')