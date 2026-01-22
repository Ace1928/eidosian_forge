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
def test_flavor_unset_with_unexist_flavor(self):
    self.compute_sdk_client.find_flavor.side_effect = [sdk_exceptions.ResourceNotFound]
    arglist = ['--project', self.project.id, 'unexist_flavor']
    verifylist = [('project', self.project.id), ('flavor', 'unexist_flavor')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)