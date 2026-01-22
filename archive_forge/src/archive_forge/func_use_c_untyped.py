from ctypes import *
import sys
import numpy as np
from numba import _helperlib
def use_c_untyped(x):
    return c_untyped(x)