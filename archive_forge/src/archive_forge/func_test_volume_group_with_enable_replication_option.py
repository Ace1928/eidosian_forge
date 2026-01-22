from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group
def test_volume_group_with_enable_replication_option(self):
    self.volume_client.api_version = api_versions.APIVersion('3.38')
    arglist = [self.fake_volume_group.id, '--enable-replication']
    verifylist = [('group', self.fake_volume_group.id), ('enable_replication', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_groups_mock.enable_replication.assert_called_once_with(self.fake_volume_group.id)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)