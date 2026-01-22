import gc
import weakref
from numba import jit
from numba.core import types
from numba.tests.support import TestCase
import unittest
def get_impl(self, dispatcher):
    """
        Get the single implementation (a C function object) of a dispatcher.
        """
    self.assertEqual(len(dispatcher.overloads), 1)
    cres = list(dispatcher.overloads.values())[0]
    return cres.entry_point