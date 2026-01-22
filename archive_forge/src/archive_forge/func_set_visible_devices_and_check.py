import multiprocessing
import os
from numba.core import config
from numba.cuda.cudadrv.runtime import runtime
from numba.cuda.testing import unittest, SerialMixin, skip_on_cudasim
from unittest.mock import patch
def set_visible_devices_and_check(q):
    try:
        from numba import cuda
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        q.put(len(cuda.gpus.lst))
    except:
        q.put(-1)