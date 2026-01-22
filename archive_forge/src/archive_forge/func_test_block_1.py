import pickle
import threading
from .. import errors, osutils, tests
from ..tests import features
def test_block_1(self):
    calls = []

    def profiled():
        calls.append('profiled')

    def do_profile():
        lsprof.profile(profiled)
        calls.append('after_profiled')
    thread = threading.Thread(target=do_profile)
    lsprof.BzrProfiler.profiler_lock.acquire()
    try:
        try:
            thread.start()
        finally:
            lsprof.BzrProfiler.profiler_lock.release()
    finally:
        thread.join()
    self.assertLength(2, calls)