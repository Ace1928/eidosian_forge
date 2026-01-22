from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import local_ip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_local_ips_delete(self):
    arglist = []
    for a in self._local_ips:
        arglist.append(a.name)
    verifylist = [('local_ip', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for a in self._local_ips:
        calls.append(call(a))
    self.network_client.delete_local_ip.assert_has_calls(calls)
    self.assertIsNone(result)