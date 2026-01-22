from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import test_components as tc
from fire import testutils
def testTestComponents(self):
    self.assertIsNotNone(tc.Empty)
    self.assertIsNotNone(tc.OldStyleEmpty)