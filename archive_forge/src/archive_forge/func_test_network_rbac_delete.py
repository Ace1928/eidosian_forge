from unittest import mock
from unittest.mock import call
import ddt
from osc_lib import exceptions
from openstackclient.network.v2 import network_rbac
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_network_rbac_delete(self):
    arglist = [self.rbac_policies[0].id]
    verifylist = [('rbac_policy', [self.rbac_policies[0].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.find_rbac_policy.assert_called_once_with(self.rbac_policies[0].id, ignore_missing=False)
    self.network_client.delete_rbac_policy.assert_called_once_with(self.rbac_policies[0])
    self.assertIsNone(result)