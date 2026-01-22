import atexit
import binascii
import functools
import hashlib
import operator
import os
import time
import numpy
import warnings
from numpy.linalg import LinAlgError
import cupy
from cupy import _core
from cupy import cuda
from cupy.cuda import curand
from cupy.cuda import device
from cupy.random import _kernels
from cupy import _util
import cupyx
def set_random_state(rs):
    """Sets the state of the random number generator for the current device.

    Args:
        state(RandomState): Random state to set for the current device.
    """
    if not isinstance(rs, RandomState):
        raise TypeError('Random state must be an instance of RandomState. Actual: {}'.format(type(rs)))
    _random_states[device.get_device_id()] = rs