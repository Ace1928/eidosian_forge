from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import messages as osc_messages
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_message_delete_exception(self):
    arglist = [self.message.id]
    verifylist = [('message', [self.message.id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.messages_mock.delete.side_effect = exceptions.CommandError()
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)