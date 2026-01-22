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
def testFirePartialNamedArgs(self):
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '1', '2']), (1, 2))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha', '1', '2']), (1, 2))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--beta', '1', '2']), (2, 1))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '1', '--alpha', '2']), (2, 1))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '1', '--beta', '2']), (1, 2))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha', '1', '--beta', '2']), (1, 2))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--beta', '1', '--alpha', '2']), (2, 1))