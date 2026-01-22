import importlib
import importlib.util
import inspect
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from PIL import Image
import pytest
import matplotlib as mpl
from matplotlib import _c_internal_utils
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.testing import subprocess_run_helper as _run_helper
@pytest.mark.skipif(sys.platform == 'win32', reason='No other signal available to send on Windows')
@pytest.mark.parametrize('env', _get_testable_interactive_backends())
@pytest.mark.parametrize('target, kwargs', [('show', {'block': True}), ('pause', {'interval': 10})])
def test_other_signal_before_sigint(env, target, kwargs, request):
    backend = env.get('MPLBACKEND')
    if not backend.startswith(('qt', 'macosx')):
        pytest.skip('SIGINT currently only tested on qt and macosx')
    if backend == 'macosx':
        request.node.add_marker(pytest.mark.xfail(reason='macosx backend is buggy'))
    proc = _WaitForStringPopen([sys.executable, '-c', inspect.getsource(_test_other_signal_before_sigint_impl) + f'\n_test_other_signal_before_sigint_impl({backend!r}, {target!r}, {kwargs!r})'])
    try:
        proc.wait_for('DRAW')
        os.kill(proc.pid, signal.SIGUSR1)
        proc.wait_for('SIGUSR1')
        os.kill(proc.pid, signal.SIGINT)
        stdout, _ = proc.communicate(timeout=_test_timeout)
    except Exception:
        proc.kill()
        stdout, _ = proc.communicate()
        raise
    print(stdout)
    assert 'SUCCESS' in stdout