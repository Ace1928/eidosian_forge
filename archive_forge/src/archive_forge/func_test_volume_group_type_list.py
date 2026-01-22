from unittest import mock
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group_type
def test_volume_group_type_list(self):
    self.volume_client.api_version = api_versions.APIVersion('3.11')
    arglist = []
    verifylist = [('show_default', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_group_types_mock.list.assert_called_once_with()
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(tuple(self.data), data)