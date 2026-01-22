from unittest import mock
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group_type
def test_volume_group_type_unset(self):
    self.volume_client.api_version = api_versions.APIVersion('3.11')
    arglist = [self.fake_volume_group_type.id, '--property', 'fizz']
    verifylist = [('group_type', self.fake_volume_group_type.id), ('properties', ['fizz'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_group_types_mock.get.assert_has_calls([mock.call(self.fake_volume_group_type.id), mock.call(self.fake_volume_group_type.id)])
    self.fake_volume_group_type.unset_keys.assert_called_once_with(['fizz'])
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)