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
def test_private_flavor_show(self):
    private_flavor = compute_fakes.create_one_flavor(attrs={'os-flavor-access:is_public': False})
    self.compute_sdk_client.find_flavor.return_value = private_flavor
    arglist = [private_flavor.name]
    verifylist = [('flavor', private_flavor.name)]
    data_with_project = (private_flavor.is_disabled, private_flavor.ephemeral, [self.flavor_access.tenant_id], private_flavor.description, private_flavor.disk, private_flavor.id, private_flavor.name, private_flavor.is_public, format_columns.DictColumn(private_flavor.extra_specs), private_flavor.ram, private_flavor.rxtx_factor, private_flavor.swap, private_flavor.vcpus)
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.get_flavor_access.assert_called_with(flavor=private_flavor.id)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(data_with_project, data)