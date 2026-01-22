from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_name_and_enable(self):
    arglist = ['--name', 'new_network_flavor', '--enable', self.new_network_flavor.name]
    verifylist = [('name', 'new_network_flavor'), ('enable', True), ('flavor', self.new_network_flavor.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'name': 'new_network_flavor', 'enabled': True}
    self.network_client.update_flavor.assert_called_with(self.new_network_flavor, **attrs)
    self.assertIsNone(result)