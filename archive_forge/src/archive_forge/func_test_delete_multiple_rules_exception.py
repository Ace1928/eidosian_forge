from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_meter_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_delete_multiple_rules_exception(self):
    arglist = [self.rule_list[0].id, 'xxxx-yyyy-zzzz', self.rule_list[1].id]
    verifylist = [('meter_rule_id', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    return_find = [self.rule_list[0], exceptions.NotFound('404'), self.rule_list[1]]
    self.network_client.find_metering_label_rule = mock.Mock(side_effect=return_find)
    ret_delete = [None, exceptions.NotFound('404')]
    self.network_client.delete_metering_label_rule = mock.Mock(side_effect=ret_delete)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    calls = [call(self.rule_list[0]), call(self.rule_list[1])]
    self.network_client.delete_metering_label_rule.assert_has_calls(calls)