from unittest import mock
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import server_group
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
def test_server_group_list_with_all_projects_and_long_v264(self, sm_mock):
    arglist = ['--all-projects', '--long']
    verifylist = [('all_projects', True), ('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.server_groups.assert_called_once_with(all_projects=True)
    self.assertCountEqual(self.list_columns_v264_long, columns)
    self.assertCountEqual(self.list_data_v264_long, tuple(data))