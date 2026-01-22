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
@with_multiprocessing
@skipif(sys.platform != 'win32' or (), reason='PermissionError only easily triggerable on Windows')
def test_resource_tracker_retries_when_permissionerror(tmpdir):
    filename = tmpdir.join('test.mmap').strpath
    cmd = 'if 1:\n    import os\n    import numpy as np\n    import time\n    from joblib.externals.loky.backend import resource_tracker\n    resource_tracker.VERBOSE = 1\n\n    # Start the resource tracker\n    resource_tracker.ensure_running()\n    time.sleep(1)\n\n    # Create a file containing numpy data\n    memmap = np.memmap(r"{filename}", dtype=np.float64, shape=10, mode=\'w+\')\n    memmap[:] = np.arange(10).astype(np.int8).data\n    memmap.flush()\n    assert os.path.exists(r"{filename}")\n    del memmap\n\n    # Create a np.memmap backed by this file\n    memmap = np.memmap(r"{filename}", dtype=np.float64, shape=10, mode=\'w+\')\n    resource_tracker.register(r"{filename}", "file")\n\n    # Ask the resource_tracker to delete the file backing the np.memmap , this\n    # should raise PermissionError that the resource_tracker will log.\n    resource_tracker.maybe_unlink(r"{filename}", "file")\n\n    # Wait for the resource_tracker to process the maybe_unlink before cleaning\n    # up the memmap\n    time.sleep(2)\n    '.format(filename=filename)
    p = subprocess.Popen([sys.executable, '-c', cmd], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    p.wait()
    out, err = p.communicate()
    assert p.returncode == 0
    assert out == b''
    msg = 'tried to unlink {}, got PermissionError'.format(filename)
    assert msg in err.decode()