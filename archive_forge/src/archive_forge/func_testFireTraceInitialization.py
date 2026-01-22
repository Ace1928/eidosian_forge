from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import testutils
from fire import trace
def testFireTraceInitialization(self):
    t = trace.FireTrace(10)
    self.assertIsNotNone(t)
    self.assertIsNotNone(t.elements)