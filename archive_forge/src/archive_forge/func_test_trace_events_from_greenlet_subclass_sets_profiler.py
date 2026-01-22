from __future__ import print_function
import sys
import greenlet
import unittest
from . import TestCase
from . import PY312
def test_trace_events_from_greenlet_subclass_sets_profiler(self):
    tracer = PythonTracer()

    class X(greenlet.greenlet):

        def run(self):
            tracer.__enter__()
            return tpt_callback()
    self._check_trace_events_from_greenlet_sets_profiler(X(), tracer)