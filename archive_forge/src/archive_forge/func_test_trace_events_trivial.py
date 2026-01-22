from __future__ import print_function
import sys
import greenlet
import unittest
from . import TestCase
from . import PY312
def test_trace_events_trivial(self):
    with PythonTracer() as actions:
        tpt_callback()
    self.assertEqual(actions, [('return', '__enter__'), ('call', 'tpt_callback'), ('return', 'tpt_callback'), ('call', '__exit__'), ('c_call', '__exit__')])