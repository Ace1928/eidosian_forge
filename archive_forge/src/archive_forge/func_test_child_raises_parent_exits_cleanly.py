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
@parametrize('backend', [pytest.param('multiprocessing', marks=pytest.mark.xfail(reason='https://github.com/joblib/joblib/issues/1086')), 'loky'])
def test_child_raises_parent_exits_cleanly(backend):
    cmd = 'if 1:\n        import os\n        from pathlib import Path\n        from time import sleep\n\n        import numpy as np\n        from joblib import Parallel, delayed\n        from testutils import print_filename_and_raise\n\n        data = np.random.rand(1000)\n\n        def get_temp_folder(parallel_obj, backend):\n            if "{b}" == "loky":\n                return Path(parallel_obj._backend._workers._temp_folder)\n            else:\n                return Path(parallel_obj._backend._pool._temp_folder)\n\n\n        if __name__ == "__main__":\n            try:\n                with Parallel(n_jobs=2, backend="{b}", max_nbytes=100) as p:\n                    temp_folder = get_temp_folder(p, "{b}")\n                    p(delayed(print_filename_and_raise)(data)\n                              for i in range(1))\n            except ValueError as e:\n                # the temporary folder should be deleted by the end of this\n                # call but apparently on some file systems, this takes\n                # some time to be visible.\n                #\n                # We attempt to write into the temporary folder to test for\n                # its existence and we wait for a maximum of 10 seconds.\n                for i in range(100):\n                    try:\n                        with open(temp_folder / "some_file.txt", "w") as f:\n                            f.write("some content")\n                    except FileNotFoundError:\n                        # temp_folder has been deleted, all is fine\n                        break\n\n                    # ... else, wait a bit and try again\n                    sleep(.1)\n                else:\n                    raise AssertionError(\n                        str(temp_folder) + " was not deleted"\n                    ) from e\n    '.format(b=backend)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.dirname(__file__)
    p = subprocess.Popen([sys.executable, '-c', cmd], stderr=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    p.wait()
    out, err = p.communicate()
    out, err = (out.decode(), err.decode())
    filename = out.split('\n')[0]
    assert p.returncode == 0, err or out
    assert err == ''
    assert not os.path.exists(filename)