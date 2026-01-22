from ctypes import *
import sys
import numpy as np
from numba import _helperlib
def use_ctype_wrapping(x):
    return ctype_wrapping(x)