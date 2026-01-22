from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group
def test_volume_group_list(self):
    self.volume_client.api_version = api_versions.APIVersion('3.13')
    arglist = ['--all-projects']
    verifylist = [('all_projects', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_groups_mock.list.assert_called_once_with(search_opts={'all_tenants': True})
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(tuple(self.data), data)