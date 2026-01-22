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
def check_ipc_array(self, index_arg=None, foreign=False):
    arr = np.arange(10, dtype=np.intp)
    devarr = cuda.to_device(arr)
    if index_arg is not None:
        devarr = devarr[index_arg]
    if foreign:
        devarr = cuda.as_cuda_array(ForeignArray(devarr))
    expect = devarr.copy_to_host()
    ipch = devarr.get_ipc_handle()
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    args = (ipch, result_queue)
    proc = ctx.Process(target=ipc_array_test, args=args)
    proc.start()
    succ, out = result_queue.get()
    if not succ:
        self.fail(out)
    else:
        np.testing.assert_equal(expect, out)
    proc.join(3)