from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_qos_policy_delete(self):
    arglist = [self.new_rule.qos_policy_id, self.new_rule.id]
    verifylist = [('qos_policy', self.new_rule.qos_policy_id), ('id', self.new_rule.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.find_qos_policy.assert_called_once_with(self.qos_policy.id, ignore_missing=False)
    self.network_client.delete_qos_bandwidth_limit_rule.assert_called_once_with(self.new_rule.id, self.qos_policy.id)
    self.assertIsNone(result)