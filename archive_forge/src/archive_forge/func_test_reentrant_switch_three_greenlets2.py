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
def test_reentrant_switch_three_greenlets2(self):
    output = self.run_script('fail_switch_three_greenlets2.py')
    self.assertIn("RESULTS: [('trace', 'switch'), ('trace', 'switch'), ('g2 arg', 'g2 from tracefunc'), ('trace', 'switch'), ('main g1', 'from g2_run'), ('trace', 'switch'), ('g1 arg', 'g1 from main'), ('trace', 'switch'), ('main g2', 'from g1_run'), ('trace', 'switch'), ('g1 from parent', 'g1 from main 2'), ('trace', 'switch'), ('main g1.2', 'g1 done'), ('trace', 'switch'), ('g2 from parent', ()), ('trace', 'switch'), ('main g2.2', 'g2 done')]", output)