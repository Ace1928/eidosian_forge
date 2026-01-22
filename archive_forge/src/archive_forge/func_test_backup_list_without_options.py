from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
def test_backup_list_without_options(self):
    arglist = []
    verifylist = [('long', False), ('name', None), ('status', None), ('volume', None), ('marker', None), ('limit', None), ('all_projects', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_sdk_client.find_volume.assert_not_called()
    self.volume_sdk_client.find_backup.assert_not_called()
    self.volume_sdk_client.backups.assert_called_with(name=None, status=None, volume_id=None, all_tenants=False, marker=None, limit=None)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))