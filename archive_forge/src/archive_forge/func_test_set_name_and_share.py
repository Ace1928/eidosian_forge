from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import address_scope
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_name_and_share(self):
    arglist = ['--name', 'new_address_scope', '--share', self._address_scope.name]
    verifylist = [('name', 'new_address_scope'), ('share', True), ('address_scope', self._address_scope.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'name': 'new_address_scope', 'shared': True}
    self.network_client.update_address_scope.assert_called_with(self._address_scope, **attrs)
    self.assertIsNone(result)