import gc
import weakref
from numba import jit
from numba.core import types
from numba.tests.support import TestCase
import unittest
@njit
def is_point_in_polygons(point, polygons):
    num_polygons = polygons.shape[0]
    if num_polygons != 0:
        intentionally_unused_variable = polygons[0]
    return 0