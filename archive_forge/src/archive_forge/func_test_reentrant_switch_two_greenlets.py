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
def test_reentrant_switch_two_greenlets(self):
    output = self.run_script('fail_switch_two_greenlets.py')
    self.assertIn('In g1_run', output)
    self.assertIn('TRACE', output)
    self.assertIn('LEAVE TRACE', output)
    self.assertIn('Falling off end of main', output)
    self.assertIn('Falling off end of g1_run', output)
    self.assertIn('Falling off end of g2', output)