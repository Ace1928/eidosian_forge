from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import test_components as tc
from fire import testutils
from fire import trace
import mock
import six
def testHelpWithNamespaceCollision(self):
    with self.assertOutputMatches(stdout='DESCRIPTION.*', stderr=None):
        core.Fire(tc.WithHelpArg, command=['--help', 'False'])
    with self.assertOutputMatches(stdout='help in a dict', stderr=None):
        core.Fire(tc.WithHelpArg, command=['dictionary', '__help'])
    with self.assertOutputMatches(stdout='{}', stderr=None):
        core.Fire(tc.WithHelpArg, command=['dictionary', '--help'])
    with self.assertOutputMatches(stdout='False', stderr=None):
        core.Fire(tc.function_with_help, command=['False'])