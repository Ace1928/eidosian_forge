from contextlib import contextmanager
import numpy as np
from numba import cuda
from numba.cuda.testing import (unittest, skip_on_cudasim,
from numba.tests.support import captured_stderr
from numba.core import config
def test_pinned_contextmanager(self):

    class PinnedException(Exception):
        pass
    arr = np.zeros(1)
    ctx = cuda.current_context()
    ctx.deallocations.clear()
    with self.check_ignored_exception(ctx):
        with cuda.pinned(arr):
            pass
        with cuda.pinned(arr):
            pass
        with cuda.defer_cleanup():
            with cuda.pinned(arr):
                pass
            with cuda.pinned(arr):
                pass
        try:
            with cuda.pinned(arr):
                raise PinnedException
        except PinnedException:
            with cuda.pinned(arr):
                pass