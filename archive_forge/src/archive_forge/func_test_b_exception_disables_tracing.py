from __future__ import print_function
import sys
import greenlet
import unittest
from . import TestCase
from . import PY312
def test_b_exception_disables_tracing(self):
    main = greenlet.getcurrent()

    def dummy():
        main.switch()
    g = greenlet.greenlet(dummy)
    g.switch()
    with GreenletTracer(error_on_trace=True) as actions:
        self.assertRaises(SomeError, g.switch)
        self.assertEqual(greenlet.gettrace(), None)
    self.assertEqual(actions, [('switch', (main, g))])