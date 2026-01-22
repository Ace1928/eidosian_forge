import os
import numpy as np
from nipype.interfaces.base import Undefined
import nipype.interfaces.fsl.maths as fsl
from nipype.interfaces.fsl import no_fsl
import pytest
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_erosion(create_files_in_directory_plus_output_type):
    files, testdir, out_ext = create_files_in_directory_plus_output_type
    erode = fsl.ErodeImage(in_file='a.nii', out_file='b.nii')
    assert erode.cmd == 'fslmaths'
    assert erode.cmdline == 'fslmaths a.nii -ero b.nii'
    erode.inputs.minimum_filter = True
    assert erode.cmdline == 'fslmaths a.nii -eroF b.nii'
    erode = fsl.ErodeImage(in_file='a.nii')
    assert erode.cmdline == 'fslmaths a.nii -ero {}'.format(os.path.join(testdir, 'a_ero{}'.format(out_ext)))