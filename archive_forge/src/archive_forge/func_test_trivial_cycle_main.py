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
def test_trivial_cycle_main(self):
    with self.assertRaises(AttributeError) as exc:
        greenlet.getcurrent().parent = greenlet.getcurrent()
    self.assertEqual(str(exc.exception), 'cannot set the parent of a main greenlet')