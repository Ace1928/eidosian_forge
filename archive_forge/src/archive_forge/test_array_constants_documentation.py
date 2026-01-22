import numpy as np
import unittest
from numba import njit
from numba.core.errors import TypingError
from numba import jit, typeof
from numba.core import types
from numba.tests.support import TestCase

        Test issue https://github.com/numba/numba/issues/2188 where freezing
        a constant array into the code that's prohibitively long and consumes
        too much RAM.
        