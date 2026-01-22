from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet_pool
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_set_address_scope(self):
    arglist = ['--address-scope', self._address_scope.id, self._subnet_pool.name]
    verifylist = [('address_scope', self._address_scope.id), ('subnet_pool', self._subnet_pool.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'address_scope_id': self._address_scope.id}
    self.network_client.update_subnet_pool.assert_called_once_with(self._subnet_pool, **attrs)
    self.assertIsNone(result)