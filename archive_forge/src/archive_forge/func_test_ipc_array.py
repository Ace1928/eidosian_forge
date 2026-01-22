import multiprocessing as mp
import itertools
import traceback
import pickle
import numpy as np
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import (skip_on_arm, skip_on_cudasim,
from numba.tests.support import linux_only, windows_only
import unittest
def test_ipc_array(self):
    for device_num in range(len(cuda.gpus)):
        arr = np.random.random(10)
        devarr = cuda.to_device(arr)
        ipch = devarr.get_ipc_handle()
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        args = (ipch, device_num, result_queue)
        proc = ctx.Process(target=staged_ipc_array_test, args=args)
        proc.start()
        succ, out = result_queue.get()
        proc.join(3)
        if not succ:
            self.fail(out)
        else:
            np.testing.assert_equal(arr, out)