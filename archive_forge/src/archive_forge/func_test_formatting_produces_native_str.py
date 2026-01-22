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
def test_formatting_produces_native_str(self):
    g_dead = RawGreenlet(lambda: None)
    g_not_started = RawGreenlet(lambda: None)
    g_cur = greenlet.getcurrent()
    for g in (g_dead, g_not_started, g_cur):
        self.assertIsInstance('%s' % (g,), str)
        self.assertIsInstance('%r' % (g,), str)