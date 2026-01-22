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
def testFireKeywordArgsWithMissingPositionalArgs(self):
    self.assertEqual(fire.Fire(tc.Kwargs, command=['run', 'Hello', 'World', '--cell', 'is']), ('Hello', 'World', {'cell': 'is'}))
    self.assertEqual(fire.Fire(tc.Kwargs, command=['run', 'Hello', '--cell', 'ok']), ('Hello', None, {'cell': 'ok'}))