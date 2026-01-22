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
def test_unexpected_reparenting(self):
    another = []

    def worker():
        g = RawGreenlet(lambda: None)
        another.append(g)
        g.switch()
    t = threading.Thread(target=worker)
    t.start()
    t.join(10)
    self.wait_for_pending_cleanups(initial_main_greenlets=self.main_greenlets_before_test + 1)

    class convoluted(RawGreenlet):

        def __getattribute__(self, name):
            if name == 'run':
                self.parent = another[0]
            return RawGreenlet.__getattribute__(self, name)
    g = convoluted(lambda: None)
    with self.assertRaises(greenlet.error) as exc:
        g.switch()
    self.assertEqual(str(exc.exception), 'cannot switch to a different thread (which happens to have exited)')
    del another[:]