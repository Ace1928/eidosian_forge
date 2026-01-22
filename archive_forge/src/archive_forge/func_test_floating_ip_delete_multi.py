from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip as fip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_floating_ip_delete_multi(self):
    self.network_client.find_ip.side_effect = [self.floating_ips[0], self.floating_ips[1]]
    arglist = []
    verifylist = []
    for f in self.floating_ips:
        arglist.append(f.id)
    verifylist = [('floating_ip', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = [call(self.floating_ips[0].id, ignore_missing=False), call(self.floating_ips[1].id, ignore_missing=False)]
    self.network_client.find_ip.assert_has_calls(calls)
    calls = []
    for f in self.floating_ips:
        calls.append(call(f))
    self.network_client.delete_ip.assert_has_calls(calls)
    self.assertIsNone(result)