from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet_pool
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_subnet_pool_list_name(self):
    subnet_pool = network_fakes.FakeSubnetPool.create_one_subnet_pool()
    self.network_client.find_network = mock.Mock(return_value=subnet_pool)
    arglist = ['--name', subnet_pool.name]
    verifylist = [('name', subnet_pool.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    filters = {'name': subnet_pool.name}
    self.network_client.subnet_pools.assert_called_once_with(**filters)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))