import sys
import os
import ctypes
import weakref
import functools
import warnings
import logging
import threading
import asyncio
import pathlib
from itertools import product
from abc import ABCMeta, abstractmethod
from ctypes import (c_int, byref, c_size_t, c_char, c_char_p, addressof,
import contextlib
import importlib
import numpy as np
from collections import namedtuple, deque
from numba import mviewbuf
from numba.core import utils, serialize, config
from .error import CudaSupportError, CudaDriverError
from .drvapi import API_PROTOTYPES
from .drvapi import cu_occupancy_b2d_size, cu_stream_callback_pyobj, cu_uuid
from numba.cuda.cudadrv import enums, drvapi, nvrtc, _extras
def pop_active_context(self):
    """Pop the active CUDA context and return the handle.
        If no CUDA context is active, return None.
        """
    with self.get_active_context() as ac:
        if ac.devnum is not None:
            if USE_NV_BINDING:
                return driver.cuCtxPopCurrent()
            else:
                popped = drvapi.cu_context()
                driver.cuCtxPopCurrent(byref(popped))
                return popped