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
@parametrize('backend', ['multiprocessing', 'loky'])
def test_resource_tracker_silent_when_reference_cycles(backend):
    if backend == 'loky' and sys.platform.startswith('win'):
        pytest.xfail('The temporary folder cannot be deleted on Windows in the presence of a reference cycle')
    cmd = 'if 1:\n        import numpy as np\n        from joblib import Parallel, delayed\n\n\n        data = np.random.rand(int(2e6)).reshape((int(1e6), 2))\n\n        # Build a complex cyclic reference that is likely to delay garbage\n        # collection of the memmapped array in the worker processes.\n        first_list = current_list = [data]\n        for i in range(10):\n            current_list = [current_list]\n        first_list.append(current_list)\n\n        if __name__ == "__main__":\n            results = Parallel(n_jobs=2, backend="{b}")(\n                delayed(len)(current_list) for i in range(10))\n            assert results == [1] * 10\n    '.format(b=backend)
    p = subprocess.Popen([sys.executable, '-c', cmd], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    p.wait()
    out, err = p.communicate()
    out = out.decode()
    err = err.decode()
    assert p.returncode == 0, out + '\n\n' + err
    assert 'resource_tracker' not in err, err