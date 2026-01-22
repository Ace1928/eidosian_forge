import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_threaded_access_property(self):
    called = collections.deque()

    class A(object):

        @misc.cachedproperty
        def b(self):
            called.append(1)
            time.sleep(random.random() * 0.5)
            return 'b'
    a = A()
    threads = []
    try:
        for _i in range(0, 20):
            t = threading_utils.daemon_thread(lambda: a.b)
            threads.append(t)
        for t in threads:
            t.start()
    finally:
        while threads:
            t = threads.pop()
            t.join()
    self.assertEqual(1, len(called))
    self.assertEqual('b', a.b)