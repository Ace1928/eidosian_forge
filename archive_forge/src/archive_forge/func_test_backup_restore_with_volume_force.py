from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
def test_backup_restore_with_volume_force(self):
    arglist = ['--force', self.backup.id, self.volume.name]
    verifylist = [('force', True), ('backup', self.backup.id), ('volume', self.volume.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.volume_sdk_client.restore_backup.assert_called_with(self.backup.id, volume_id=self.volume.id, name=None)
    self.assertIsNotNone(result)