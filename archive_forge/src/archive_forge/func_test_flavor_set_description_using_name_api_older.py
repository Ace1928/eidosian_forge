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
def test_flavor_set_description_using_name_api_older(self):
    arglist = ['--description', 'description', self.flavor.name]
    verifylist = [('description', 'description'), ('flavor', self.flavor.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch.object(sdk_utils, 'supports_microversion', return_value=False):
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)