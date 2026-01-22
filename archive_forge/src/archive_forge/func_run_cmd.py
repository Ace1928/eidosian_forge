import os
import sys
import subprocess
import threading
from numba import cuda
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
from numba.tests.support import captured_stdout
def run_cmd(self, cmdline, env):
    popen = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    timeout = threading.Timer(5 * 60.0, popen.kill)
    try:
        timeout.start()
        out, err = popen.communicate()
        return (out.decode(), err.decode())
    finally:
        timeout.cancel()
    return (None, None)