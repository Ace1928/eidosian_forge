import os
import numpy as np
from nipype.interfaces.base import Undefined
import nipype.interfaces.fsl.maths as fsl
from nipype.interfaces.fsl import no_fsl
import pytest
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_tempfilt(create_files_in_directory_plus_output_type):
    files, testdir, out_ext = create_files_in_directory_plus_output_type
    filt = fsl.TemporalFilter(in_file='a.nii', out_file='b.nii')
    assert filt.cmd == 'fslmaths'
    assert filt.cmdline == 'fslmaths a.nii -bptf -1.000000 -1.000000 b.nii'
    windows = [(-1, -1), (0.1, 0.1), (-1, 20), (20, -1), (128, 248)]
    for win in windows:
        filt.inputs.highpass_sigma = win[0]
        filt.inputs.lowpass_sigma = win[1]
        assert filt.cmdline == 'fslmaths a.nii -bptf {:.6f} {:.6f} b.nii'.format(win[0], win[1])
    filt = fsl.TemporalFilter(in_file='a.nii', highpass_sigma=64)
    assert filt.cmdline == 'fslmaths a.nii -bptf 64.000000 -1.000000 {}'.format(os.path.join(testdir, 'a_filt{}'.format(out_ext)))