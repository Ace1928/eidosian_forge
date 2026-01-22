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
def test_lazy_lock_init(self):
    for meth in ('fork', 'spawn', 'forkserver'):
        try:
            multiprocessing.get_context(meth)
        except ValueError:
            continue
        cmd = "import numba; import multiprocessing;multiprocessing.set_start_method('{}');print(multiprocessing.get_context().get_start_method())"
        cmdline = [sys.executable, '-c', cmd.format(meth)]
        out, err = self.run_cmd(cmdline)
        if self._DEBUG:
            print('OUT:', out)
            print('ERR:', err)
        self.assertIn(meth, out)