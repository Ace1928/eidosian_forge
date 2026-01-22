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
def test_exc_state(self):

    def f():
        try:
            raise ValueError('fun')
        except:
            exc_info = sys.exc_info()
            RawGreenlet(h).switch()
            self.assertEqual(exc_info, sys.exc_info())

    def h():
        self.assertEqual(sys.exc_info(), (None, None, None))
    RawGreenlet(f).switch()