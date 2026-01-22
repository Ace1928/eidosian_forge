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
def test_namesource(setup_file):
    tmp_infile = setup_file
    tmpd, nme, ext = split_filename(tmp_infile)

    class spec2(nib.CommandLineInputSpec):
        moo = nib.File(name_source=['doo'], hash_files=False, argstr='%s', position=2)
        doo = nib.File(exists=True, argstr='%s', position=1)
        goo = traits.Int(argstr='%d', position=4)
        poo = nib.File(name_source=['goo'], hash_files=False, argstr='%s', position=3)

    class TestName(nib.CommandLine):
        _cmd = 'mycommand'
        input_spec = spec2
    testobj = TestName()
    testobj.inputs.doo = tmp_infile
    testobj.inputs.goo = 99
    assert '%s_generated' % nme in testobj.cmdline
    assert '%d_generated' % testobj.inputs.goo in testobj.cmdline
    testobj.inputs.moo = 'my_%s_template'
    assert 'my_%s_template' % nme in testobj.cmdline