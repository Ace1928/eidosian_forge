from __future__ import print_function
import sys
import greenlet
import unittest
from . import TestCase
from . import PY312
@unittest.skipIf(*DEBUG_BUILD_PY312)
def test_trace_events_multiple_greenlets_switching(self):
    tracer = PythonTracer()
    g1 = None
    g2 = None

    def g1_run():
        tracer.__enter__()
        tpt_callback()
        g2.switch()
        tpt_callback()
        return 42

    def g2_run():
        tpt_callback()
        tracer.__exit__()
        tpt_callback()
        g1.switch()
    g1 = greenlet.greenlet(g1_run)
    g2 = greenlet.greenlet(g2_run)
    x = g1.switch()
    self.assertEqual(x, 42)
    tpt_callback()
    self.assertEqual(tracer.actions, [('return', '__enter__'), ('call', 'tpt_callback'), ('return', 'tpt_callback'), ('c_call', 'g1_run'), ('call', 'g2_run'), ('call', 'tpt_callback'), ('return', 'tpt_callback'), ('call', '__exit__'), ('c_call', '__exit__')])