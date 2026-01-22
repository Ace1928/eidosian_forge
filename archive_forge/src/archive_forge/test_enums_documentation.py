import numpy as np
from numba import int8, int16, int32
from numba import cuda, vectorize, njit
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.enum_usecases import (

Test cases adapted from numba/tests/test_enums.py
