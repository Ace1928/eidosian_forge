from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet_pool
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_set_with_default_quota(self):
    arglist = ['--default-quota', '20', self._subnet_pool.name]
    verifylist = [('default_quota', 20), ('subnet_pool', self._subnet_pool.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.update_subnet_pool.assert_called_once_with(self._subnet_pool, **{'default_quota': 20})
    self.assertIsNone(result)