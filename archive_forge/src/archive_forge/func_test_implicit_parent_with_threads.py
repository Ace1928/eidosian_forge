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
def test_implicit_parent_with_threads(self):
    if not gc.isenabled():
        return
    N = gc.get_threshold()[0]
    if N < 50:
        return

    def attempt():
        lock1 = threading.Lock()
        lock1.acquire()
        lock2 = threading.Lock()
        lock2.acquire()
        recycled = [False]

        def another_thread():
            lock1.acquire()
            greenlet.getcurrent()
            lock2.release()
        t = threading.Thread(target=another_thread)
        t.start()

        class gc_callback(object):

            def __del__(self):
                lock1.release()
                lock2.acquire()
                recycled[0] = True

        class garbage(object):

            def __init__(self):
                self.cycle = self
                self.callback = gc_callback()
        l = []
        x = range(N * 2)
        current = greenlet.getcurrent()
        g = garbage()
        for _ in x:
            g = None
            if recycled[0]:
                t.join(10)
                return False
            last = RawGreenlet()
            if recycled[0]:
                break
            l.append(last)
        else:
            gc.collect()
            if recycled[0]:
                t.join(10)
            return False
        self.assertEqual(last.parent, current)
        for g in l:
            self.assertEqual(g.parent, current)
        return True
    for _ in range(5):
        if attempt():
            break