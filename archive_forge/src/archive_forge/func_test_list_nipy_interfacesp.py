import pytest
import sys
from contextlib import contextmanager
from io import StringIO
from ...utils import nipype_cmd
def test_list_nipy_interfacesp(self):
    with pytest.raises(SystemExit) as cm:
        with capture_sys_output() as (stdout, stderr):
            nipype_cmd.main(['nipype_cmd', 'nipype.interfaces.nipy'])
    with pytest.raises(SystemExit) as cm:
        with capture_sys_output() as (stdout, stderr):
            nipype_cmd.main(['nipype_cmd', 'nipype.interfaces.nipy'])
    exit_exception = cm.value
    assert exit_exception.code == 0
    assert stderr.getvalue() == ''
    assert stdout.getvalue() == 'Available Interfaces:\n\tComputeMask\n\tEstimateContrast\n\tFitGLM\n\tSimilarity\n\tSpaceTimeRealigner\n'