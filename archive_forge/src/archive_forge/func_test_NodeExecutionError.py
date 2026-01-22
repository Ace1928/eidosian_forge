import os
from copy import deepcopy
import pytest
from .... import config
from ....interfaces import utility as niu
from ....interfaces import base as nib
from ... import engine as pe
from ..utils import merge_dict
from .test_base import EngineTestInterface
from .test_utils import UtilsTestInterface
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
import nipype.interfaces.spm as spm
import os
from io import StringIO
from nipype.utils.config import config
def test_NodeExecutionError(tmp_path, monkeypatch):
    import stat
    monkeypatch.chdir(tmp_path)
    exebin = tmp_path / 'bin'
    exebin.mkdir()
    exe = exebin / 'nipype-node-execution-fail'
    exe.write_text('#!/bin/bash\necho "Running"\necho "This should fail" >&2\nexit 1', encoding='utf-8')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv('PATH', str(exe.parent.absolute()), prepend=os.pathsep)
    cmd = pe.Node(FailCommandLine(), name='cmd-fail', base_dir='cmd')
    with pytest.raises(pe.nodes.NodeExecutionError) as exc:
        cmd.run()
    error_msg = str(exc.value)
    for attr in ('Cmdline:', 'Stdout:', 'Stderr:', 'Traceback:'):
        assert attr in error_msg
    assert 'This should fail' in error_msg

    def fail():
        raise Exception('Functions can fail too')
    func = pe.Node(niu.Function(function=fail), name='func-fail', base_dir='func')
    with pytest.raises(pe.nodes.NodeExecutionError) as exc:
        func.run()
    error_msg = str(exc.value)
    assert 'Traceback:' in error_msg
    assert 'Cmdline:' not in error_msg
    assert 'Functions can fail too' in error_msg