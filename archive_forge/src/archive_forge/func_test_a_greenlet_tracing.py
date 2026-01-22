from __future__ import print_function
import sys
import greenlet
import unittest
from . import TestCase
from . import PY312
def test_a_greenlet_tracing(self):
    main = greenlet.getcurrent()

    def dummy():
        pass

    def dummyexc():
        raise SomeError()
    with GreenletTracer() as actions:
        g1 = greenlet.greenlet(dummy)
        g1.switch()
        g2 = greenlet.greenlet(dummyexc)
        self.assertRaises(SomeError, g2.switch)
    self.assertEqual(actions, [('switch', (main, g1)), ('switch', (g1, main)), ('switch', (main, g2)), ('throw', (g2, main))])