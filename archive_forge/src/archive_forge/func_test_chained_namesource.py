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
def test_chained_namesource(setup_file):
    tmp_infile = setup_file
    tmpd, nme, ext = split_filename(tmp_infile)

    class spec2(nib.CommandLineInputSpec):
        doo = nib.File(exists=True, argstr='%s', position=1)
        moo = nib.File(name_source=['doo'], hash_files=False, argstr='%s', position=2, name_template='%s_mootpl')
        poo = nib.File(name_source=['moo'], hash_files=False, argstr='%s', position=3)

    class TestName(nib.CommandLine):
        _cmd = 'mycommand'
        input_spec = spec2
    testobj = TestName()
    testobj.inputs.doo = tmp_infile
    res = testobj.cmdline
    assert '%s' % tmp_infile in res
    assert '%s_mootpl ' % nme in res
    assert '%s_mootpl_generated' % nme in res