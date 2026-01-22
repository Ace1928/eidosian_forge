import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
def test_CommandLine_output(tmpdir):
    tmpdir.chdir()
    file = tmpdir.join('foo.txt')
    file.write('123456\n')
    name = os.path.basename(file.strpath)
    ci = nib.CommandLine(command='ls -l')
    ci.terminal_output = 'allatonce'
    res = ci.run()
    assert res.runtime.merged == ''
    assert name in res.runtime.stdout
    ci = nib.CommandLine(command='ls -l')
    ci.terminal_output = 'file_stdout'
    res = ci.run()
    assert os.path.isfile('stdout.nipype')
    assert name in res.runtime.stdout
    tmpdir.join('stdout.nipype').remove(ignore_errors=True)
    ci = nib.CommandLine(command='ls -l')
    ci.terminal_output = 'file_stderr'
    res = ci.run()
    assert os.path.isfile('stderr.nipype')
    tmpdir.join('stderr.nipype').remove(ignore_errors=True)
    ci = nib.CommandLine(command='ls -l')
    ci.terminal_output = 'none'
    res = ci.run()
    assert res.runtime.stdout == '' and res.runtime.stderr == '' and (res.runtime.merged == '')
    ci = nib.CommandLine(command='ls -l')
    res = ci.run()
    assert ci.terminal_output == 'stream'
    assert name in res.runtime.stdout and res.runtime.stderr == ''
    ci = nib.CommandLine(command='ls -l')
    ci.terminal_output = 'file'
    res = ci.run()
    assert os.path.isfile('output.nipype')
    assert name in res.runtime.merged and res.runtime.stdout == '' and (res.runtime.stderr == '')
    tmpdir.join('output.nipype').remove(ignore_errors=True)
    ci = nib.CommandLine(command='ls -l')
    ci.terminal_output = 'file_split'
    res = ci.run()
    assert os.path.isfile('stdout.nipype')
    assert os.path.isfile('stderr.nipype')
    assert name in res.runtime.stdout