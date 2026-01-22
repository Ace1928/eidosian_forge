from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_network_flavor_delete(self):
    arglist = [self._network_flavors[0].name]
    verifylist = [('flavor', [self._network_flavors[0].name])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.find_flavor.assert_called_once_with(self._network_flavors[0].name, ignore_missing=False)
    self.network_client.delete_flavor.assert_called_once_with(self._network_flavors[0])
    self.assertIsNone(result)