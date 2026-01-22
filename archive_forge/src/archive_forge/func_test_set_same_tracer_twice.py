from __future__ import print_function
import sys
import greenlet
import unittest
from . import TestCase
from . import PY312
def test_set_same_tracer_twice(self):
    tracer = GreenletTracer()
    with tracer:
        greenlet.settrace(tracer)