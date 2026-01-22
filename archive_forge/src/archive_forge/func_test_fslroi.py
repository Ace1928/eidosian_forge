import os
import numpy as np
import nibabel as nb
import pytest
import nipype.interfaces.fsl.utils as fsl
from nipype.interfaces.fsl import no_fsl, Info
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_fslroi(create_files_in_directory_plus_output_type):
    filelist, outdir, _ = create_files_in_directory_plus_output_type
    roi = fsl.ExtractROI()
    assert roi.cmd == 'fslroi'
    with pytest.raises(ValueError):
        roi.run()
    roi.inputs.in_file = filelist[0]
    roi.inputs.roi_file = 'foo_roi.nii'
    roi.inputs.t_min = 10
    roi.inputs.t_size = 20
    assert roi.cmdline == 'fslroi %s foo_roi.nii 10 20' % filelist[0]
    roi2 = fsl.ExtractROI(in_file=filelist[0], roi_file='foo2_roi.nii', t_min=20, t_size=40, x_min=3, x_size=30, y_min=40, y_size=10, z_min=5, z_size=20)
    assert roi2.cmdline == 'fslroi %s foo2_roi.nii 3 30 40 10 5 20 20 40' % filelist[0]