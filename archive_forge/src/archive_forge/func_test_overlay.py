import os
import numpy as np
import nibabel as nb
import pytest
import nipype.interfaces.fsl.utils as fsl
from nipype.interfaces.fsl import no_fsl, Info
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_overlay(create_files_in_directory_plus_output_type):
    filelist, outdir, _ = create_files_in_directory_plus_output_type
    overlay = fsl.Overlay()
    assert overlay.cmd == 'overlay'
    with pytest.raises(ValueError):
        overlay.run()
    overlay.inputs.stat_image = filelist[0]
    overlay.inputs.stat_thresh = (2.5, 10)
    overlay.inputs.background_image = filelist[1]
    overlay.inputs.auto_thresh_bg = True
    overlay.inputs.show_negative_stats = True
    overlay.inputs.out_file = 'foo_overlay.nii'
    assert overlay.cmdline == 'overlay 1 0 %s -a %s 2.50 10.00 %s -2.50 -10.00 foo_overlay.nii' % (filelist[1], filelist[0], filelist[0])
    overlay2 = fsl.Overlay(stat_image=filelist[0], stat_thresh=(2.5, 10), background_image=filelist[1], auto_thresh_bg=True, out_file='foo2_overlay.nii')
    assert overlay2.cmdline == 'overlay 1 0 %s -a %s 2.50 10.00 foo2_overlay.nii' % (filelist[1], filelist[0])