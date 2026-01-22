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
def test_check_threading_layer_is_gnu(self):
    runme = "if 1:\n            from numba.np.ufunc import omppool\n            assert omppool.openmp_vendor == 'GNU'\n            "
    cmdline = [sys.executable, '-c', runme]
    out, err = self.run_cmd(cmdline)