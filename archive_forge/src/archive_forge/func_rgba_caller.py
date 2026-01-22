import re
import types
import numpy as np
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from numba import cuda, jit, float32, int32
from numba.core.errors import TypingError
@cuda.jit
def rgba_caller(x, channels):
    x[0] = rgba(channels[0], channels[1], channels[2], channels[3])