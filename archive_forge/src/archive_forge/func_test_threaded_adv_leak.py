from __future__ import print_function, absolute_import, division
import sys
import gc
import time
import weakref
import threading
import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
from .leakcheck import ignores_leakcheck
from .leakcheck import RUNNING_ON_MANYLINUX
def test_threaded_adv_leak(self):
    gg = []

    def worker():
        ll = greenlet.getcurrent().ll = []

        def additional():
            ll.append(greenlet.getcurrent())
        for _ in range(2):
            greenlet.greenlet(additional).switch()
        gg.append(weakref.ref(greenlet.getcurrent()))
    for _ in range(2):
        t = threading.Thread(target=worker)
        t.start()
        t.join(10)
        del t
    greenlet.getcurrent()
    self.__recycle_threads()
    greenlet.getcurrent()
    gc.collect()
    greenlet.getcurrent()
    for g in gg:
        self.assertIsNone(g())