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
def test_unexpected_reparenting_thread_running(self):
    another = []
    switched_to_greenlet = threading.Event()
    keep_main_alive = threading.Event()

    def worker():
        g = RawGreenlet(lambda: None)
        another.append(g)
        g.switch()
        switched_to_greenlet.set()
        keep_main_alive.wait(10)

    class convoluted(RawGreenlet):

        def __getattribute__(self, name):
            if name == 'run':
                self.parent = another[0]
            return RawGreenlet.__getattribute__(self, name)
    t = threading.Thread(target=worker)
    t.start()
    switched_to_greenlet.wait(10)
    try:
        g = convoluted(lambda: None)
        with self.assertRaises(greenlet.error) as exc:
            g.switch()
        self.assertEqual(str(exc.exception), 'cannot switch to a different thread')
    finally:
        keep_main_alive.set()
        t.join(10)
        del another[:]