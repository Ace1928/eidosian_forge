from unittest import mock
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import server_group
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
def test_server_group_create_with_soft_policies_pre_v215(self, sm_mock):
    arglist = ['--policy', 'soft-anti-affinity', 'affinity_group']
    verifylist = [('policy', 'soft-anti-affinity'), ('name', 'affinity_group')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-compute-api-version 2.15 or greater is required', str(ex))