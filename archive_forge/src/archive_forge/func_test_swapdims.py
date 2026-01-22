import os
import numpy as np
import nibabel as nb
import pytest
import nipype.interfaces.fsl.utils as fsl
from nipype.interfaces.fsl import no_fsl, Info
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_swapdims(create_files_in_directory_plus_output_type):
    files, testdir, out_ext = create_files_in_directory_plus_output_type
    swap = fsl.SwapDimensions()
    assert swap.cmd == 'fslswapdim'
    args = [dict(in_file=files[0]), dict(new_dims=('x', 'y', 'z'))]
    for arg in args:
        wontrun = fsl.SwapDimensions(**arg)
        with pytest.raises(ValueError):
            wontrun.run()
    swap.inputs.in_file = files[0]
    swap.inputs.new_dims = ('x', 'y', 'z')
    assert swap.cmdline == 'fslswapdim a.nii x y z %s' % os.path.realpath(os.path.join(testdir, 'a_newdims%s' % out_ext))
    swap.inputs.out_file = 'b.nii'
    assert swap.cmdline == 'fslswapdim a.nii x y z b.nii'