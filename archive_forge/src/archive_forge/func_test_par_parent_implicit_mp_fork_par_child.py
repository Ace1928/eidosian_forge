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
def test_par_parent_implicit_mp_fork_par_child(self):
    """
        Implicit use of multiprocessing fork context.
        Does this:
        1. Start with OpenMP
        2. Fork to processes using OpenMP (this is invalid)
        3. Joins fork
        4. Check the exception pushed onto the queue that is a result of
           catching SIGTERM coming from the C++ aborting on illegal fork
           pattern for GNU OpenMP
        """
    body = 'if 1:\n            mp = multiprocessing.get_context(\'fork\')\n            X = np.arange(1000000.)\n            Y = np.arange(1000000.)\n            q = mp.Queue()\n\n            # Start OpenMP runtime on parent via parallel function\n            Z = busy_func(X, Y, q)\n\n            # fork() underneath with no exec, will abort\n            proc = mp.Process(target = busy_func, args=(X, Y, q))\n            proc.start()\n\n            err = q.get()\n            assert "Caught SIGTERM" in str(err)\n        '
    runme = self.template % body
    cmdline = [sys.executable, '-c', runme]
    out, err = self.run_cmd(cmdline)
    if self._DEBUG:
        print(out, err)