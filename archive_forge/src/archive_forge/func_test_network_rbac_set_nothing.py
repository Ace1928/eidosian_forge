from unittest import mock
from unittest.mock import call
import ddt
from osc_lib import exceptions
from openstackclient.network.v2 import network_rbac
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_network_rbac_set_nothing(self):
    arglist = [self.rbac_policy.id]
    verifylist = [('rbac_policy', self.rbac_policy.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.find_rbac_policy.assert_called_once_with(self.rbac_policy.id, ignore_missing=False)
    attrs = {}
    self.network_client.update_rbac_policy.assert_called_once_with(self.rbac_policy, **attrs)
    self.assertIsNone(result)