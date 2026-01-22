from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_security_group_rules_delete(self):
    arglist = []
    verifylist = []
    for s in self._security_group_rules:
        arglist.append(s.id)
    verifylist = [('rule', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for s in self._security_group_rules:
        calls.append(call(s))
    self.network_client.delete_security_group_rule.assert_has_calls(calls)
    self.assertIsNone(result)