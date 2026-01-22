from unittest import mock
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import server_group
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
def test_server_group_list_with_offset(self, sm_mock):
    arglist = ['--offset', '5']
    verifylist = [('all_projects', False), ('long', False), ('limit', None), ('offset', 5)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.compute_sdk_client.server_groups.assert_called_once_with(offset=5)