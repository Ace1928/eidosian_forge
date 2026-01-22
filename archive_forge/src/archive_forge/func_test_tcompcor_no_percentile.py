import os
import nibabel as nb
import numpy as np
import pytest
from ...testing import utils
from ..confounds import CompCor, TCompCor, ACompCor
def test_tcompcor_no_percentile(self):
    ccinterface = TCompCor(num_components=6, realigned_file=self.realigned_file)
    ccinterface.run()
    mask = nb.load('mask_000.nii.gz').dataobj
    num_nonmasked_voxels = np.count_nonzero(mask)
    assert num_nonmasked_voxels == 1