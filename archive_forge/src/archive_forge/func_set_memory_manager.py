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
def set_memory_manager(mm_plugin):
    """Configure Numba to use an External Memory Management (EMM) Plugin. If
    the EMM Plugin version does not match one supported by this version of
    Numba, a RuntimeError will be raised.

    :param mm_plugin: The class implementing the EMM Plugin.
    :type mm_plugin: BaseCUDAMemoryManager
    :return: None
    """
    global _memory_manager
    dummy = mm_plugin(context=None)
    iv = dummy.interface_version
    if iv != _SUPPORTED_EMM_INTERFACE_VERSION:
        err = 'EMM Plugin interface has version %d - version %d required' % (iv, _SUPPORTED_EMM_INTERFACE_VERSION)
        raise RuntimeError(err)
    _memory_manager = mm_plugin