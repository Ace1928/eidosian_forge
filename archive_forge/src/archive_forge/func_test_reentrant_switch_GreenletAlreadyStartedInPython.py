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
def test_reentrant_switch_GreenletAlreadyStartedInPython(self):
    output = self.run_script('fail_initialstub_already_started.py')
    self.assertIn("RESULTS: ['Begin C', 'Switch to b from B.__getattribute__ in C', ('Begin B', ()), '_B_run switching to main', ('main from c', 'From B'), 'B.__getattribute__ back from main in C', ('Begin A', (None,)), ('A dead?', True, 'B dead?', True, 'C dead?', False), 'C done', ('main from c.2', None)]", output)