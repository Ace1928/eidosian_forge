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
def test_cycle_namesource2(setup_file):
    tmp_infile = setup_file
    tmpd, nme, ext = split_filename(tmp_infile)

    class spec3(nib.CommandLineInputSpec):
        moo = nib.File(name_source=['doo'], hash_files=False, argstr='%s', position=1, name_template='%s_mootpl')
        poo = nib.File(name_source=['moo'], hash_files=False, argstr='%s', position=2)
        doo = nib.File(name_source=['poo'], hash_files=False, argstr='%s', position=3)

    class TestCycle(nib.CommandLine):
        _cmd = 'mycommand'
        input_spec = spec3
    to1 = TestCycle()
    to1.inputs.poo = tmp_infile
    not_raised = True
    try:
        res = to1.cmdline
    except nib.NipypeInterfaceError:
        not_raised = False
    print(res)
    assert not_raised
    assert '%s' % tmp_infile in res
    assert '%s_generated' % nme in res
    assert '%s_generated_mootpl' % nme in res