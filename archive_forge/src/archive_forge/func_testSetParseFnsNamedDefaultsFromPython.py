from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import decorators
from fire import testutils
def testSetParseFnsNamedDefaultsFromPython(self):
    self.assertTupleEqual(WithDefaults().example2(), (10, int))
    self.assertEqual(WithDefaults().example2(5), (5, int))
    self.assertEqual(WithDefaults().example2(12.0), (12, float))