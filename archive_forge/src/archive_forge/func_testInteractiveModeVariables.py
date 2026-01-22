from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import test_components as tc
from fire import testutils
from fire import trace
import mock
import six
@mock.patch('fire.interact.Embed')
def testInteractiveModeVariables(self, mock_embed):
    core.Fire(tc.WithDefaults, command=['double', '2', '--', '-i'])
    self.assertTrue(mock_embed.called)
    (variables, verbose), unused_kwargs = mock_embed.call_args
    self.assertFalse(verbose)
    self.assertEqual(variables['result'], 4)
    self.assertIsInstance(variables['self'], tc.WithDefaults)
    self.assertIsInstance(variables['trace'], trace.FireTrace)