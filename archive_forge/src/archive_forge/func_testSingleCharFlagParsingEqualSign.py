from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import fire
from fire import test_components as tc
from fire import testutils
import mock
import six
def testSingleCharFlagParsingEqualSign(self):
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-a=True']), (True, '0'))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-a=3', '--beta=10']), (3, 10))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-a=False', '-b=15']), (False, 15))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-a', '42', '-b=12']), (42, 12))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-a=42', '-b', '10']), (42, 10))