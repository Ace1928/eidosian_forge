import numpy as np
from numba.core.errors import TypingError
from numba import njit
from numba.core import types
import struct
import unittest

        Test error due mishandling of Optional to Optional casting

        Related issue: https://github.com/numba/numba/issues/1718
        