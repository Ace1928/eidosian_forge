import numpy as np
from numba import typeof, njit
from numba.tests.support import MemoryLeakMixin
import unittest

        A bug breaks array return if the function starts with a loop
        