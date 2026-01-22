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
def test_reentrant_switch_three_greenlets(self):
    ex = self.assertScriptRaises('fail_switch_three_greenlets.py', exitcodes=(1,))
    self.assertIn('TypeError', ex.output)
    self.assertIn('positional arguments', ex.output)