from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_network_flavors_delete(self):
    arglist = []
    verifylist = []
    for a in self._network_flavors:
        arglist.append(a.name)
    verifylist = [('flavor', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for a in self._network_flavors:
        calls.append(mock.call(a))
    self.network_client.delete_flavor.assert_has_calls(calls)
    self.assertIsNone(result)