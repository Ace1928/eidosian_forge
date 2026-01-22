import gc
import multiprocessing
import os
import pickle
import pytest
from rpy2 import rinterface
import rpy2
import rpy2.rinterface_lib._rinterface_capi as _rinterface
import signal
import sys
import subprocess
import tempfile
import textwrap
import time
@pytest.mark.parametrize('rcode', ('while(TRUE) {}', '\n     i <- 0;\n     while(TRUE) {\n       i <- i+1;\n       Sys.sleep(0.01);\n     }\n     '))
def test_interrupt_r(rcode):
    expected_code = 42
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as rpy_code:
        rpy2_path = os.path.dirname(rpy2.__path__[0])
        rpy_code_str = textwrap.dedent("\n        import sys\n        sys.path.insert(0, '%s')\n        import rpy2.rinterface as ri\n        from rpy2.rinterface_lib import callbacks\n        from rpy2.rinterface_lib import embedded\n\n        ri.initr()\n        def f(x):\n            # This flush is important to make sure we avoid a deadlock.\n            print(x, flush=True)\n        rcode = '''\n        message('executing-rcode')\n        console.flush()\n        %s\n        '''\n        with callbacks.obj_in_module(callbacks, 'consolewrite_print', f):\n            try:\n                ri.baseenv['eval'](ri.parse(rcode))\n            except embedded.RRuntimeError:\n                sys.exit(%d)\n      ") % (rpy2_path, rcode, expected_code)
        rpy_code.write(rpy_code_str)
    cmd = (sys.executable, rpy_code.name)
    with open(os.devnull, 'w') as fnull:
        creationflags = 0
        if os.name == 'nt':
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=fnull, creationflags=creationflags) as child_proc:
            for line in child_proc.stdout:
                if line == b'executing-rcode\n':
                    break
            sigint = signal.CTRL_C_EVENT if os.name == 'nt' else signal.SIGINT
            child_proc.send_signal(sigint)
            ret_code = child_proc.wait(timeout=10)
    assert ret_code == expected_code