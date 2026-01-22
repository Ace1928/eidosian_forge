from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import testutils
from fire import trace
def testFireTraceElementAsStringWithTarget(self):
    el = trace.FireTraceElement(component='Example', action='Created toy', target='Beaker')
    self.assertEqual(str(el), 'Created toy "Beaker"')