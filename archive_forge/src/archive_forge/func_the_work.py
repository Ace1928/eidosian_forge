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
def the_work():
    with cuda.gpus[device_num]:
        this_ctx = cuda.devices.get_context()
        deviceptr = handle.open_staged(this_ctx)
        arrsize = handle.size // np.dtype(np.intp).itemsize
        hostarray = np.zeros(arrsize, dtype=np.intp)
        cuda.driver.device_to_host(hostarray, deviceptr, size=handle.size)
        handle.close()
    return hostarray