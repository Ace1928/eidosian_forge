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
def testFireKeywordArgs(self):
    self.assertEqual(fire.Fire(tc.Kwargs, command=['props', '--name', 'David', '--age', '24']), {'name': 'David', 'age': 24})
    self.assertEqual(fire.Fire(tc.Kwargs, command=['props', '--message', '"This is a message it has -- in it"']), {'message': 'This is a message it has -- in it'})
    self.assertEqual(fire.Fire(tc.Kwargs, command=['props', '--message', 'This is a message it has -- in it']), {'message': 'This is a message it has -- in it'})
    self.assertEqual(fire.Fire(tc.Kwargs, command='props --message "This is a message it has -- in it"'), {'message': 'This is a message it has -- in it'})
    self.assertEqual(fire.Fire(tc.Kwargs, command=['upper', '--alpha', 'A', '--beta', 'B']), 'ALPHA BETA')
    self.assertEqual(fire.Fire(tc.Kwargs, command=['upper', '--alpha', 'A', '--beta', 'B', '-', 'lower']), 'alpha beta')