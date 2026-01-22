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
@linux_only
def test_fork_from_non_main_thread(self):
    runme = "if 1:\n            import threading\n            import numba\n            numba.config.THREADING_LAYER='tbb'\n            from numba import njit, prange, objmode\n            from numba.core.serialize import PickleCallableByPath\n            import os\n\n            e_running = threading.Event()\n            e_proceed = threading.Event()\n\n            def indirect_core():\n                e_running.set()\n                # wait for forker() to have forked\n                while not e_proceed.isSet():\n                    pass\n\n            indirect = PickleCallableByPath(indirect_core)\n\n            @njit\n            def obj_mode_func():\n                with objmode():\n                    indirect()\n\n            @njit(parallel=True, nogil=True)\n            def work():\n                acc = 0\n                for x in prange(10):\n                    acc += x\n                obj_mode_func()\n                return acc\n\n            def runner():\n                work()\n\n            def forker():\n                # wait for the jit function to say it's running\n                while not e_running.isSet():\n                    pass\n                # then fork\n                os.fork()\n                # now fork is done signal the runner to proceed to exit\n                e_proceed.set()\n\n            numba_runner = threading.Thread(target=runner,)\n            fork_runner =  threading.Thread(target=forker,)\n\n            threads = (numba_runner, fork_runner)\n            for t in threads:\n                t.start()\n            for t in threads:\n                t.join()\n        "
    cmdline = [sys.executable, '-c', runme]
    out, err = self.run_cmd(cmdline)
    msg_head = 'Attempted to fork from a non-main thread, the TBB library'
    self.assertIn(msg_head, err)
    if self._DEBUG:
        print('OUT:', out)
        print('ERR:', err)