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
def core_ipc_handle_test(the_work, result_queue):
    try:
        arr = the_work()
    except:
        succ = False
        out = traceback.format_exc()
    else:
        succ = True
        out = arr
    result_queue.put((succ, out))