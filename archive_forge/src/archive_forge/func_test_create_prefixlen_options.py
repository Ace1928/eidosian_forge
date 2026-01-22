from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet_pool
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_create_prefixlen_options(self):
    arglist = ['--default-prefix-length', self._subnet_pool.default_prefixlen, '--max-prefix-length', self._subnet_pool.max_prefixlen, '--min-prefix-length', self._subnet_pool.min_prefixlen, '--pool-prefix', '10.0.10.0/24', self._subnet_pool.name]
    verifylist = [('default_prefix_length', int(self._subnet_pool.default_prefixlen)), ('max_prefix_length', int(self._subnet_pool.max_prefixlen)), ('min_prefix_length', int(self._subnet_pool.min_prefixlen)), ('name', self._subnet_pool.name), ('prefixes', ['10.0.10.0/24'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_subnet_pool.assert_called_once_with(**{'default_prefixlen': int(self._subnet_pool.default_prefixlen), 'max_prefixlen': int(self._subnet_pool.max_prefixlen), 'min_prefixlen': int(self._subnet_pool.min_prefixlen), 'prefixes': ['10.0.10.0/24'], 'name': self._subnet_pool.name})
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)