import itertools
import unittest
from numba import njit
from numba.core import types

    Test all int/float numeric conversion to ensure we have all the external
    dependencies to perform these conversions.
    