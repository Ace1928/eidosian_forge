import random
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_other_options(self):
    arglist = ['--enable', '--no-share', '--disable-port-security', self._network.name]
    verifylist = [('enable', True), ('no_share', True), ('name', self._network.name), ('external', False), ('disable_port_security', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_network.assert_called_once_with(**{'admin_state_up': True, 'name': self._network.name, 'shared': False, 'port_security_enabled': False})
    self.assertEqual(set(self.columns), set(columns))
    self.assertCountEqual(self.data, data)