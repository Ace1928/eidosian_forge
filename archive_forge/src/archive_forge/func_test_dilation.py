import os
import numpy as np
from nipype.interfaces.base import Undefined
import nipype.interfaces.fsl.maths as fsl
from nipype.interfaces.fsl import no_fsl
import pytest
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_dilation(create_files_in_directory_plus_output_type):
    files, testdir, out_ext = create_files_in_directory_plus_output_type
    diller = fsl.DilateImage(in_file='a.nii', out_file='b.nii')
    assert diller.cmd == 'fslmaths'
    with pytest.raises(ValueError):
        diller.run()
    for op in ['mean', 'modal', 'max']:
        cv = dict(mean='M', modal='D', max='F')
        diller.inputs.operation = op
        assert diller.cmdline == 'fslmaths a.nii -dil{} b.nii'.format(cv[op])
    for k in ['3D', '2D', 'box', 'boxv', 'gauss', 'sphere']:
        for size in [1, 1.5, 5]:
            diller.inputs.kernel_shape = k
            diller.inputs.kernel_size = size
            assert diller.cmdline == 'fslmaths a.nii -kernel {} {:.4f} -dilF b.nii'.format(k, size)
    f = open('kernel.txt', 'w').close()
    del f
    diller.inputs.kernel_shape = 'file'
    diller.inputs.kernel_size = Undefined
    diller.inputs.kernel_file = 'kernel.txt'
    assert diller.cmdline == 'fslmaths a.nii -kernel file kernel.txt -dilF b.nii'
    dil = fsl.DilateImage(in_file='a.nii', operation='max')
    assert dil.cmdline == 'fslmaths a.nii -dilF {}'.format(os.path.join(testdir, 'a_dil{}'.format(out_ext)))