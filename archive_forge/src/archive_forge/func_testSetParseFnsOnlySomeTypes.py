from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import decorators
from fire import testutils
def testSetParseFnsOnlySomeTypes(self):
    self.assertEqual(core.Fire(PartialParseFn, command=['example4', '10', '10']), ('10', 10))
    self.assertEqual(core.Fire(PartialParseFn, command=['example5', '10', '10']), (10, '10'))