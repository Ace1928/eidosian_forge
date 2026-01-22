import faulthandler
import itertools
import multiprocessing
import os
import random
import re
import subprocess
import sys
import textwrap
import threading
import unittest
import numpy as np
from numba import jit, vectorize, guvectorize, set_num_threads
from numba.tests.support import (temp_directory, override_config, TestCase, tag,
import queue as t_queue
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
from numba.core import config
def test_par_parent_os_fork_par_child(self):
    """
        Whilst normally valid, this actually isn't for Numba invariant of OpenMP
        Checks SIGABRT is received.
        """
    body = 'if 1:\n            X = np.arange(1000000.)\n            Y = np.arange(1000000.)\n            Z = busy_func(X, Y)\n            pid = os.fork()\n            if pid  == 0:\n                Z = busy_func(X, Y)\n            else:\n                os.wait()\n        '
    runme = self.template % body
    cmdline = [sys.executable, '-c', runme]
    try:
        out, err = self.run_cmd(cmdline)
    except AssertionError as e:
        self.assertIn('failed with code -6', str(e))