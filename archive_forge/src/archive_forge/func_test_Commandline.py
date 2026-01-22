import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
def test_Commandline():
    with pytest.raises(Exception):
        nib.CommandLine()
    ci = nib.CommandLine(command='which')
    assert ci.cmd == 'which'
    assert ci.inputs.args == nib.Undefined
    ci2 = nib.CommandLine(command='which', args='ls')
    assert ci2.cmdline == 'which ls'
    ci3 = nib.CommandLine(command='echo')
    ci3.resource_monitor = False
    ci3.inputs.environ = {'MYENV': 'foo'}
    res = ci3.run()
    assert res.runtime.environ['MYENV'] == 'foo'
    assert res.outputs is None

    class CommandLineInputSpec1(nib.CommandLineInputSpec):
        foo = nib.Str(argstr='%s', desc='a str')
        goo = nib.traits.Bool(argstr='-g', desc='a bool', position=0)
        hoo = nib.traits.List(argstr='-l %s', desc='a list')
        moo = nib.traits.List(argstr='-i %d...', desc='a repeated list', position=-1)
        noo = nib.traits.Int(argstr='-x %d', desc='an int')
        roo = nib.traits.Str(desc='not on command line')
        soo = nib.traits.Bool(argstr='-soo')
    nib.CommandLine.input_spec = CommandLineInputSpec1
    ci4 = nib.CommandLine(command='cmd')
    ci4.inputs.foo = 'foo'
    ci4.inputs.goo = True
    ci4.inputs.hoo = ['a', 'b']
    ci4.inputs.moo = [1, 2, 3]
    ci4.inputs.noo = 0
    ci4.inputs.roo = 'hello'
    ci4.inputs.soo = False
    cmd = ci4._parse_inputs()
    assert cmd[0] == '-g'
    assert cmd[-1] == '-i 1 -i 2 -i 3'
    assert 'hello' not in ' '.join(cmd)
    assert '-soo' not in ' '.join(cmd)
    ci4.inputs.soo = True
    cmd = ci4._parse_inputs()
    assert '-soo' in ' '.join(cmd)

    class CommandLineInputSpec2(nib.CommandLineInputSpec):
        foo = nib.File(argstr='%s', desc='a str', genfile=True)
    nib.CommandLine.input_spec = CommandLineInputSpec2
    ci5 = nib.CommandLine(command='cmd')
    with pytest.raises(NotImplementedError):
        ci5._parse_inputs()

    class DerivedClass(nib.CommandLine):
        input_spec = CommandLineInputSpec2

        def _gen_filename(self, name):
            return 'filename'
    ci6 = DerivedClass(command='cmd')
    assert ci6._parse_inputs()[0] == 'filename'
    nib.CommandLine.input_spec = nib.CommandLineInputSpec