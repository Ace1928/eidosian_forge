from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_qos_rule_list(self):
    arglist = [self.qos_policy.id]
    verifylist = [('qos_policy', self.qos_policy.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.find_qos_policy.assert_called_once_with(self.qos_policy.id, ignore_missing=False)
    self.assertEqual(self.columns, columns)
    list_data = list(data)
    self.assertEqual(len(self.data), len(list_data))
    for index in range(len(list_data)):
        self.assertEqual(self.data[index], list_data[index])