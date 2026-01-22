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
def testSeparatorForChaining(self):
    self.assertIsInstance(fire.Fire(tc.ReturnsObj, command=['get-obj', 'arg1', 'arg2', 'as-bool', 'True']), tc.BoolConverter)
    self.assertEqual(fire.Fire(tc.ReturnsObj, command=['get-obj', 'arg1', 'arg2', '-', 'as-bool', 'True']), True)
    self.assertEqual(fire.Fire(tc.ReturnsObj, command=['get-obj', 'arg1', 'arg2', '&', 'as-bool', 'True', '--', '--separator', '&']), True)
    self.assertEqual(fire.Fire(tc.ReturnsObj, command=['get-obj', 'arg1', '$$', 'as-bool', 'True', '--', '--separator', '$$']), True)