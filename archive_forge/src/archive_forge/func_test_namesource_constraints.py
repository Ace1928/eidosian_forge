import os
import warnings
import pytest
from ....utils.filemanip import split_filename
from ... import base as nib
from ...base import traits, Undefined
from ....interfaces import fsl
from ...utility.wrappers import Function
from ....pipeline import Node
from ..specs import get_filecopy_info
def test_namesource_constraints(setup_file):
    tmp_infile = setup_file
    tmpd, nme, ext = split_filename(tmp_infile)

    class constrained_spec(nib.CommandLineInputSpec):
        in_file = nib.File(argstr='%s', position=1)
        threshold = traits.Float(argstr='%g', xor=['mask_file'], position=2)
        mask_file = nib.File(argstr='%s', name_source=['in_file'], name_template='%s_mask', keep_extension=True, xor=['threshold'], position=2)
        out_file1 = nib.File(argstr='%s', name_source=['in_file'], name_template='%s_out1', keep_extension=True, position=3)
        out_file2 = nib.File(argstr='%s', name_source=['in_file'], name_template='%s_out2', keep_extension=True, requires=['threshold'], position=4)

    class TestConstrained(nib.CommandLine):
        _cmd = 'mycommand'
        input_spec = constrained_spec
    tc = TestConstrained()
    assert tc.cmdline == 'mycommand'
    tc.inputs.in_file = os.path.basename(tmp_infile)
    assert tc.cmdline == 'mycommand foo.txt foo_mask.txt foo_out1.txt'
    tc.inputs.threshold = 10.0
    assert tc.cmdline == 'mycommand foo.txt 10 foo_out1.txt foo_out2.txt'