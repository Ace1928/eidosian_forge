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
def test_flavor_list_private_flavors(self):
    arglist = ['--private']
    verifylist = [('public', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'is_public': False}
    self.compute_sdk_client.flavors.assert_called_with(**kwargs)
    self.compute_sdk_client.fetch_flavor_extra_specs.assert_not_called()
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, tuple(data))