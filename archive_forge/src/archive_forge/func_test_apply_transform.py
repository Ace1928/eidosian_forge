import os
import pytest
from nipype.testing import example_data
import nipype.interfaces.spm.utils as spmu
from nipype.interfaces.base import isdefined
from nipype.utils.filemanip import split_filename, fname_presuffix
from nipype.interfaces.base import TraitError
def test_apply_transform():
    moving = example_data(infile='functional.nii')
    mat = example_data(infile='trans.mat')
    applymat = spmu.ApplyTransform(matlab_cmd='mymatlab')
    assert applymat.inputs.matlab_cmd == 'mymatlab'
    applymat.inputs.in_file = moving
    applymat.inputs.mat = mat
    scrpt = applymat._make_matlab_command(None)
    expected = '[p n e v] = spm_fileparts(V.fname);'
    assert expected in scrpt
    expected = 'V.mat = transform.M * V.mat;'
    assert expected in scrpt