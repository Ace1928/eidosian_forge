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
def testFireObjectWithList(self):
    self.assertEqual(fire.Fire(tc.TypedProperties, command=['echo', '0']), 'alex')
    self.assertEqual(fire.Fire(tc.TypedProperties, command=['echo', '1']), 'bethany')