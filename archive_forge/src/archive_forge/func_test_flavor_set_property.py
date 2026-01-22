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
def test_flavor_set_property(self):
    arglist = ['--property', 'FOO="B A R"', 'baremetal']
    verifylist = [('properties', {'FOO': '"B A R"'}), ('flavor', 'baremetal')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.find_flavor.assert_called_with(parsed_args.flavor, get_extra_specs=True, ignore_missing=False)
    self.compute_sdk_client.create_flavor_extra_specs.assert_called_with(self.flavor.id, {'FOO': '"B A R"'})
    self.assertIsNone(result)