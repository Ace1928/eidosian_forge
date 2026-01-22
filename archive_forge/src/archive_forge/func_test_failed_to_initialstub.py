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
def test_failed_to_initialstub(self):

    def func():
        raise AssertionError('Never get here')
    g = greenlet._greenlet.UnswitchableGreenlet(func)
    g.force_switch_error = True
    with self.assertRaisesRegex(SystemError, 'Failed to switch stacks into a greenlet for the first time.'):
        g.switch()