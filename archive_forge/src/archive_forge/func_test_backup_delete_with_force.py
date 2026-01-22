from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
def test_backup_delete_with_force(self):
    arglist = ['--force', self.backups[0].id]
    verifylist = [('force', True), ('backups', [self.backups[0].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.volume_sdk_client.delete_backup.assert_called_with(self.backups[0].id, ignore_missing=False, force=True)
    self.assertIsNone(result)