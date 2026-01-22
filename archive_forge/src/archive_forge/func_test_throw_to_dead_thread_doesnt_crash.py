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
@fails_leakcheck
def test_throw_to_dead_thread_doesnt_crash(self):
    self._do_test_throw_to_dead_thread_doesnt_crash()