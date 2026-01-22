from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gc
import sys
import time
import threading
from abc import ABCMeta, abstractmethod
import greenlet
from greenlet import greenlet as RawGreenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def test_main_from_other_thread(self):
    main = greenlet.getcurrent()

    class T(threading.Thread):
        original_main = thread_main = None
        main_glet = None

        def run(self):
            self.original_main = repr(main)
            self.main_glet = greenlet.getcurrent()
            self.thread_main = repr(self.main_glet)
    t = T()
    t.start()
    t.join(10)
    self.assertEndsWith(t.original_main, ' suspended active started main>')
    self.assertEndsWith(t.thread_main, ' current active started main>')
    for _ in range(3):
        time.sleep(0.001)
    for _ in range(3):
        self.assertTrue(t.main_glet.dead)
        r = repr(t.main_glet)
        self.assertEndsWith(r, ' (thread exited) dead>')