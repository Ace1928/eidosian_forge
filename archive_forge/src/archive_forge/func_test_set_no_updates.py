from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import security_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_no_updates(self):
    arglist = [self._security_group.name]
    verifylist = [('group', self._security_group.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertFalse(self.network_client.update_security_group.called)
    self.assertFalse(self.network_client.set_tags.called)
    self.assertIsNone(result)