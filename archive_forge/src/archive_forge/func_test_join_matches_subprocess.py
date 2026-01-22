import pytest
import subprocess
import json
import sys
from numpy.distutils import _shell_utils
from numpy.testing import IS_WASM
@pytest.mark.skipif(IS_WASM, reason='Cannot start subprocess')
@pytest.mark.parametrize('argv', argv_cases)
def test_join_matches_subprocess(Parser, runner, argv):
    """
    Test that join produces strings understood by subprocess
    """
    cmd = [sys.executable, '-c', 'import json, sys; print(json.dumps(sys.argv[1:]))']
    joined = Parser.join(cmd + argv)
    json_out = runner(joined).decode()
    assert json.loads(json_out) == argv