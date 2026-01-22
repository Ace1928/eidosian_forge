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
def test_thread_bug(self):

    def runner(x):
        g = RawGreenlet(lambda: time.sleep(x))
        g.switch()
    t1 = threading.Thread(target=runner, args=(0.2,))
    t2 = threading.Thread(target=runner, args=(0.3,))
    t1.start()
    t2.start()
    t1.join(10)
    t2.join(10)