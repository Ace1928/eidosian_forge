from unittest import mock
from openstack.compute.v2 import flavor as _flavor
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import flavor
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_flavor_list_long_no_extra_specs(self):
    flavor = compute_fakes.create_one_flavor(attrs={'extra_specs': {}})
    self.data = ((flavor.id, flavor.name, flavor.ram, flavor.disk, flavor.ephemeral, flavor.vcpus, flavor.is_public),)
    self.data_long = (self.data[0] + (flavor.swap, flavor.rxtx_factor, format_columns.DictColumn(flavor.extra_specs)),)
    self.api_mock.side_effect = [[flavor], []]
    self.compute_sdk_client.flavors = self.api_mock
    self.compute_sdk_client.fetch_flavor_extra_specs = mock.Mock(return_value=None)
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'is_public': True}
    self.compute_sdk_client.flavors.assert_called_with(**kwargs)
    self.compute_sdk_client.fetch_flavor_extra_specs.assert_called_once_with(flavor)
    self.assertEqual(self.columns_long, columns)
    self.assertCountEqual(self.data_long, tuple(data))