import logging
import re
import sys
import warnings
from ctypes import (c_void_p, c_int, POINTER, c_char_p, c_size_t, byref,
import threading
from llvmlite import ir
from .error import NvvmError, NvvmSupportError, NvvmWarning
from .libs import get_libdevice, open_libdevice, open_cudalib
from numba.core import cgutils, config
def find_closest_arch(mycc):
    """
    Given a compute capability, return the closest compute capability supported
    by the CUDA toolkit.

    :param mycc: Compute capability as a tuple ``(MAJOR, MINOR)``
    :return: Closest supported CC as a tuple ``(MAJOR, MINOR)``
    """
    supported_ccs = NVVM().supported_ccs
    if not supported_ccs:
        msg = 'No supported GPU compute capabilities found. Please check your cudatoolkit version matches your CUDA version.'
        raise NvvmSupportError(msg)
    for i, cc in enumerate(supported_ccs):
        if cc == mycc:
            return cc
        elif cc > mycc:
            if i == 0:
                msg = 'GPU compute capability %d.%d is not supported(requires >=%d.%d)' % (mycc + cc)
                raise NvvmSupportError(msg)
            else:
                return supported_ccs[i - 1]
    return supported_ccs[-1]