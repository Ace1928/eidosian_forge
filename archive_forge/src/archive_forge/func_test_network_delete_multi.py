from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_network_delete_multi(self, net_mock):
    net_mock.return_value = mock.Mock(return_value=None)
    arglist = []
    for n in self._networks:
        arglist.append(n['id'])
    verifylist = [('network', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for n in self._networks:
        calls.append(call(n['id']))
    net_mock.assert_has_calls(calls)
    self.assertIsNone(result)