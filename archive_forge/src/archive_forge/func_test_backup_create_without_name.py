from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
def test_backup_create_without_name(self):
    arglist = ['--description', self.new_backup.description, '--container', self.new_backup.container, self.new_backup.volume_id]
    verifylist = [('description', self.new_backup.description), ('container', self.new_backup.container), ('volume', self.new_backup.volume_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_sdk_client.create_backup.assert_called_with(volume_id=self.new_backup.volume_id, container=self.new_backup.container, name=None, description=self.new_backup.description, force=False, incremental=False)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)