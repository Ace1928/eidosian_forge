import os
import gc
import sys
from joblib._multiprocessing_helpers import mp
from joblib.testing import SkipTest, skipif
def with_memory_profiler(func):
    """A decorator to skip tests requiring memory_profiler."""

    def dummy_func():
        raise SkipTest('Test requires memory_profiler.')
    return dummy_func