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
def testBoolParsingSingleHyphen(self):
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-alpha=False', '10']), (False, 10))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-alpha', '-beta', '10']), (True, 10))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-alpha', '-beta=10']), (True, 10))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-noalpha', '-beta']), (False, True))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-alpha', '-10', '-beta']), (-10, True))