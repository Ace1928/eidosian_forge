import os
import pytest
from nipype.testing import example_data
import nipype.interfaces.spm.utils as spmu
from nipype.interfaces.base import isdefined
from nipype.utils.filemanip import split_filename, fname_presuffix
from nipype.interfaces.base import TraitError
def test_reslice():
    moving = example_data(infile='functional.nii')
    space_defining = example_data(infile='T1.nii')
    reslice = spmu.Reslice(matlab_cmd='mymatlab_version')
    assert reslice.inputs.matlab_cmd == 'mymatlab_version'
    reslice.inputs.in_file = moving
    reslice.inputs.space_defining = space_defining
    assert reslice.inputs.interp == 0
    with pytest.raises(TraitError):
        reslice.inputs.trait_set(interp='nearest')
    with pytest.raises(TraitError):
        reslice.inputs.trait_set(interp=10)
    reslice.inputs.interp = 1
    script = reslice._make_matlab_command(None)
    outfile = fname_presuffix(moving, prefix='r')
    assert reslice.inputs.out_file == outfile
    expected = '\nflags.mean=0;\nflags.which=1;\nflags.mask=0;'
    assert expected in script.replace(' ', '')
    expected_interp = 'flags.interp = 1;\n'
    assert expected_interp in script
    assert 'spm_reslice(invols, flags);' in script