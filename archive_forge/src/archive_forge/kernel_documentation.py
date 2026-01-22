from contextlib import contextmanager
import functools
import sys
import threading
import numpy as np
from .cudadrv.devicearray import FakeCUDAArray, FakeWithinKernelCUDAArray
from .kernelapi import Dim3, FakeCUDAModule, swapped_cuda_module
from ..errors import normalize_kernel_dimensions
from ..args import wrap_arg, ArgHint

    Manages the execution of a thread block.

    When run() is called, all threads are started. Each thread executes until it
    hits syncthreads(), at which point it sets its own syncthreads_blocked to
    True so that the BlockManager knows it is blocked. It then waits on its
    syncthreads_event.

    The BlockManager polls threads to determine if they are blocked in
    syncthreads(). If it finds a blocked thread, it adds it to the set of
    blocked threads. When all threads are blocked, it unblocks all the threads.
    The thread are unblocked by setting their syncthreads_blocked back to False
    and setting their syncthreads_event.

    The polling continues until no threads are alive, when execution is
    complete.
    