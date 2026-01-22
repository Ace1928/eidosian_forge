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
def cache_config(self, prefer_equal=False, prefer_cache=False, prefer_shared=False):
    prefer_equal = prefer_equal or (prefer_cache and prefer_shared)
    attr = binding.CUfunction_attribute
    if prefer_equal:
        flag = attr.CU_FUNC_CACHE_PREFER_EQUAL
    elif prefer_cache:
        flag = attr.CU_FUNC_CACHE_PREFER_L1
    elif prefer_shared:
        flag = attr.CU_FUNC_CACHE_PREFER_SHARED
    else:
        flag = attr.CU_FUNC_CACHE_PREFER_NONE
    driver.cuFuncSetCacheConfig(self.handle, flag)