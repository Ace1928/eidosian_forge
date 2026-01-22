from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_security_group_rule_show_all_options(self):
    arglist = [self._security_group_rule['id']]
    verifylist = [('rule', self._security_group_rule['id'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_client.api.security_group_list.assert_called_once_with()
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)