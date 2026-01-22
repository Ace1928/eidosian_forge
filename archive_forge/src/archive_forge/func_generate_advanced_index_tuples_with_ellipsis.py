import itertools
import numpy as np
import unittest
from numba import jit, typeof, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin, TestCase
def generate_advanced_index_tuples_with_ellipsis(self, N, maxdim, many=True):
    """
        Same as generate_advanced_index_tuples(), but also insert an
        ellipsis at various points.
        """
    for tup in self.generate_advanced_index_tuples(N, maxdim, many):
        for i in range(len(tup) + 1):
            yield (tup[:i] + (Ellipsis,) + tup[i:])