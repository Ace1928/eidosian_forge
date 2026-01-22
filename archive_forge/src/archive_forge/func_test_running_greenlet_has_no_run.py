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
def test_running_greenlet_has_no_run(self):
    has_run = []

    def func():
        has_run.append(hasattr(greenlet.getcurrent(), 'run'))
    g = RawGreenlet(func)
    g.switch()
    self.assertEqual(has_run, [False])