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
def methgen(impl, p):

    def test_method(self):
        selfproc = multiprocessing.current_process()
        if selfproc.daemon:
            _msg = 'daemonized processes cannot have children'
            self.skipTest(_msg)
        else:
            self.run_compile(impl, parallelism=p)
    return test_method