from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import address_scope
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_no_share(self):
    arglist = ['--no-share', self.new_address_scope.name]
    verifylist = [('no_share', True), ('name', self.new_address_scope.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_address_scope.assert_called_once_with(**{'ip_version': self.new_address_scope.ip_version, 'shared': False, 'name': self.new_address_scope.name})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)