from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import decorators
from fire import testutils
def testSetParseFnsDefaultsFromFire(self):
    self.assertEqual(core.Fire(WithDefaults, command=['example1']), (10, int))
    self.assertEqual(core.Fire(WithDefaults, command=['example1', '10']), (10, float))
    self.assertEqual(core.Fire(WithDefaults, command=['example1', '13']), (13, float))
    self.assertEqual(core.Fire(WithDefaults, command=['example1', '14.0']), (14, float))