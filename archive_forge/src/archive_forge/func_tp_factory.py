import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, tag
import unittest
def tp_factory(tp_name):
    return getattr(np, tp_name)