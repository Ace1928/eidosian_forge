import sys
import unittest
from numba.tests.support import captured_stdout
from numba.core.config import IS_WIN32
def make_singlethread(inner_func):
    """
                Run the given function inside a single thread.
                """

    def func(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        inner_func(result, *args)
        return result
    return func