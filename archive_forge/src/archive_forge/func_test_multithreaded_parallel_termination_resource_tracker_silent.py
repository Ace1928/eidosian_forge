import os
import mmap
import sys
import platform
import gc
import pickle
import itertools
from time import sleep
import subprocess
import threading
import faulthandler
import pytest
from joblib.test.common import with_numpy, np
from joblib.test.common import with_multiprocessing
from joblib.test.common import with_dev_shm
from joblib.testing import raises, parametrize, skipif
from joblib.backports import make_memmap
from joblib.parallel import Parallel, delayed
from joblib.pool import MemmappingPool
from joblib.executor import _TestingMemmappingExecutor as TestExecutor
from joblib._memmapping_reducer import has_shareable_memory
from joblib._memmapping_reducer import ArrayMemmapForwardReducer
from joblib._memmapping_reducer import _strided_from_memmap
from joblib._memmapping_reducer import _get_temp_dir
from joblib._memmapping_reducer import _WeakArrayKeyMap
from joblib._memmapping_reducer import _get_backing_memmap
import joblib._memmapping_reducer as jmr
@with_numpy
@with_multiprocessing
def test_multithreaded_parallel_termination_resource_tracker_silent():
    cmd = 'if 1:\n        import os\n        import numpy as np\n        from joblib import Parallel, delayed\n        from joblib.externals.loky.backend import resource_tracker\n        from concurrent.futures import ThreadPoolExecutor, wait\n\n        resource_tracker.VERBOSE = 0\n\n        array = np.arange(int(1e2))\n\n        temp_dirs_thread_1 = set()\n        temp_dirs_thread_2 = set()\n\n\n        def raise_error(array):\n            raise ValueError\n\n\n        def parallel_get_filename(array, temp_dirs):\n            with Parallel(backend="loky", n_jobs=2, max_nbytes=10) as p:\n                for i in range(10):\n                    [filename] = p(\n                        delayed(getattr)(array, "filename") for _ in range(1)\n                    )\n                    temp_dirs.add(os.path.dirname(filename))\n\n\n        def parallel_raise(array, temp_dirs):\n            with Parallel(backend="loky", n_jobs=2, max_nbytes=10) as p:\n                for i in range(10):\n                    [filename] = p(\n                        delayed(raise_error)(array) for _ in range(1)\n                    )\n                    temp_dirs.add(os.path.dirname(filename))\n\n\n        executor = ThreadPoolExecutor(max_workers=2)\n\n        # both function calls will use the same loky executor, but with a\n        # different Parallel object.\n        future_1 = executor.submit({f1}, array, temp_dirs_thread_1)\n        future_2 = executor.submit({f2}, array, temp_dirs_thread_2)\n\n        # Wait for both threads to terminate their backend\n        wait([future_1, future_2])\n\n        future_1.result()\n        future_2.result()\n    '
    functions_and_returncodes = [('parallel_get_filename', 'parallel_get_filename', 0), ('parallel_get_filename', 'parallel_raise', 1), ('parallel_raise', 'parallel_raise', 1)]
    for f1, f2, returncode in functions_and_returncodes:
        p = subprocess.Popen([sys.executable, '-c', cmd.format(f1=f1, f2=f2)], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        p.wait()
        out, err = p.communicate()
        assert p.returncode == returncode, out.decode()
        assert b'resource_tracker' not in err, err.decode()