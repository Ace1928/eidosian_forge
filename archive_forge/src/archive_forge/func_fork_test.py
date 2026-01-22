import os
import multiprocessing as mp
import numpy as np
from numba import cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
def fork_test(q):
    from numba.cuda.cudadrv.error import CudaDriverError
    try:
        cuda.to_device(np.arange(1))
    except CudaDriverError as e:
        q.put(e)
    else:
        q.put(None)