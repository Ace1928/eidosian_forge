import ctypes
import ctypes.util
import os
import sys
import threading
import warnings
import numpy as np
import unittest
from numba import jit
from numba.core import errors
from numba.tests.support import TestCase, tag
def lifted_f(a, indices):
    """
    Same as f(), but inside a lifted loop
    """
    object()
    for idx in indices:
        sleep(10 * sleep_factor)
        a[idx] = PyThread_get_thread_ident()