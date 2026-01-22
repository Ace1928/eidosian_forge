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
def testInteractiveModeFullArgument(self, mock_embed):
    core.Fire(tc.TypedProperties, command=['alpha', '--', '--interactive'])
    self.assertTrue(mock_embed.called)