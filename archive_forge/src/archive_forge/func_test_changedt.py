import os
import numpy as np
from nipype.interfaces.base import Undefined
import nipype.interfaces.fsl.maths as fsl
from nipype.interfaces.fsl import no_fsl
import pytest
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_changedt(create_files_in_directory_plus_output_type):
    files, testdir, out_ext = create_files_in_directory_plus_output_type
    cdt = fsl.ChangeDataType()
    assert cdt.cmd == 'fslmaths'
    with pytest.raises(ValueError):
        cdt.run()
    cdt.inputs.in_file = 'a.nii'
    cdt.inputs.out_file = 'b.nii'
    with pytest.raises(ValueError):
        cdt.run()
    dtypes = ['float', 'char', 'int', 'short', 'double', 'input']
    cmdline = 'fslmaths a.nii b.nii -odt {}'
    for dtype in dtypes:
        foo = fsl.MathsCommand(in_file='a.nii', out_file='b.nii', output_datatype=dtype)
        assert foo.cmdline == cmdline.format(dtype)