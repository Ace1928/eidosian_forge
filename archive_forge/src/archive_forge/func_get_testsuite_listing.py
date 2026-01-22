import os
import sys
import subprocess
from numba import cuda
import unittest
import itertools
def get_testsuite_listing(self, args, *, subp_kwargs=None):
    """
        Use `subp_kwargs` to pass extra argument to `subprocess.check_output`.
        """
    subp_kwargs = subp_kwargs or {}
    cmd = [sys.executable, '-m', 'numba.runtests', '-l'] + list(args)
    out_bytes = subprocess.check_output(cmd, **subp_kwargs)
    lines = out_bytes.decode('UTF-8').splitlines()
    lines = [line for line in lines if line.strip()]
    return lines