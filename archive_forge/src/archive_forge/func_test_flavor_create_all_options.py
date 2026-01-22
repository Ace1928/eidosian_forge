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
def test_flavor_create_all_options(self):
    arglist = ['--id', self.flavor.id, '--ram', str(self.flavor.ram), '--disk', str(self.flavor.disk), '--ephemeral', str(self.flavor.ephemeral), '--swap', str(self.flavor.swap), '--vcpus', str(self.flavor.vcpus), '--rxtx-factor', str(self.flavor.rxtx_factor), '--public', '--description', str(self.flavor.description), '--property', 'property=value', self.flavor.name]
    verifylist = [('id', self.flavor.id), ('ram', self.flavor.ram), ('disk', self.flavor.disk), ('ephemeral', self.flavor.ephemeral), ('swap', self.flavor.swap), ('vcpus', self.flavor.vcpus), ('rxtx_factor', self.flavor.rxtx_factor), ('public', True), ('description', self.flavor.description), ('properties', {'property': 'value'}), ('name', self.flavor.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    args = {'name': self.flavor.name, 'ram': self.flavor.ram, 'vcpus': self.flavor.vcpus, 'disk': self.flavor.disk, 'id': self.flavor.id, 'ephemeral': self.flavor.ephemeral, 'swap': self.flavor.swap, 'rxtx_factor': self.flavor.rxtx_factor, 'is_public': self.flavor.is_public, 'description': self.flavor.description}
    props = {'property': 'value'}
    create_flavor = _flavor.Flavor(**self.flavor)
    expected_flavor = _flavor.Flavor(**self.flavor)
    expected_flavor.extra_specs = props
    cmp_data = list(self.data)
    cmp_data[7] = format_columns.DictColumn(props)
    self.compute_sdk_client.create_flavor.return_value = create_flavor
    self.compute_sdk_client.create_flavor_extra_specs.return_value = expected_flavor
    with mock.patch.object(sdk_utils, 'supports_microversion', return_value=True):
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.create_flavor.assert_called_once_with(**args)
        self.compute_sdk_client.create_flavor_extra_specs.assert_called_once_with(create_flavor, props)
        self.compute_sdk_client.get_flavor_access.assert_not_called()
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(tuple(cmp_data), data)