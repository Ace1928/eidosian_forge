from unittest import mock
from unittest.mock import call
import ddt
from osc_lib import exceptions
from openstackclient.network.v2 import network_rbac
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_network_policies_delete_with_exception(self):
    arglist = [self.rbac_policies[0].id, 'unexist_rbac_policy']
    verifylist = [('rbac_policy', [self.rbac_policies[0].id, 'unexist_rbac_policy'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    find_mock_result = [self.rbac_policies[0], exceptions.CommandError]
    self.network_client.find_rbac_policy = mock.Mock(side_effect=find_mock_result)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 RBAC policies failed to delete.', str(e))
    self.network_client.find_rbac_policy.assert_any_call(self.rbac_policies[0].id, ignore_missing=False)
    self.network_client.find_rbac_policy.assert_any_call('unexist_rbac_policy', ignore_missing=False)
    self.network_client.delete_rbac_policy.assert_called_once_with(self.rbac_policies[0])