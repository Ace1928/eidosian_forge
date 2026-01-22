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
def test_set_parent_wrong_types(self):

    def bg():
        greenlet.getcurrent().parent.switch()

    def check(glet):
        for p in (None, 1, self, '42'):
            with self.assertRaises(TypeError) as exc:
                glet.parent = p
            self.assertEqual(str(exc.exception), 'GreenletChecker: Expected any type of greenlet, not ' + type(p).__name__)
    g = RawGreenlet(bg)
    self.assertFalse(g)
    check(g)
    g.switch()
    self.assertTrue(g)
    check(g)
    g.switch()