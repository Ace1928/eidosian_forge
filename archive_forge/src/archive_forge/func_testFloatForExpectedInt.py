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
def testFloatForExpectedInt(self):
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '--alpha', '2.2', '--beta', '3.0']), 8.2)
    self.assertEqual(fire.Fire(tc.NumberDefaults, command=['integer_reciprocal', '--divisor', '5.0']), 0.2)
    self.assertEqual(fire.Fire(tc.NumberDefaults, command=['integer_reciprocal', '4.0']), 0.25)