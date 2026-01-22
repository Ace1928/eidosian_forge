import pickle
import threading
from .. import errors, osutils, tests
from ..tests import features
def test_start_call_stuff_stop(self):
    profiler = lsprof.BzrProfiler()
    profiler.start()
    try:

        def a_function():
            pass
        a_function()
    finally:
        stats = profiler.stop()
    stats.freeze()
    lines = [str(data) for data in stats.data]
    lines = [line for line in lines if 'a_function' in line]
    self.assertLength(1, lines)