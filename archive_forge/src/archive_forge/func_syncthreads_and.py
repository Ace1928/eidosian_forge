from contextlib import contextmanager
import functools
import sys
import threading
import numpy as np
from .cudadrv.devicearray import FakeCUDAArray, FakeWithinKernelCUDAArray
from .kernelapi import Dim3, FakeCUDAModule, swapped_cuda_module
from ..errors import normalize_kernel_dimensions
from ..args import wrap_arg, ArgHint
def syncthreads_and(self, value):
    idx = (self.threadIdx.x, self.threadIdx.y, self.threadIdx.z)
    self._manager.block_state[idx] = value
    self.syncthreads()
    test = np.all(self._manager.block_state)
    self.syncthreads()
    return 1 if test else 0