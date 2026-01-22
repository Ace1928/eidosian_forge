from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
def test_delete_multiple_backups_with_exception(self):
    arglist = [self.backups[0].id, 'unexist_backup']
    verifylist = [('backups', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    find_mock_result = [self.backups[0], exceptions.CommandError]
    self.volume_sdk_client.find_backup.side_effect = find_mock_result
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 backups failed to delete.', str(e))
    self.volume_sdk_client.find_backup.assert_any_call(self.backups[0].id, ignore_missing=False)
    self.volume_sdk_client.find_backup.assert_any_call('unexist_backup', ignore_missing=False)
    self.assertEqual(2, self.volume_sdk_client.find_backup.call_count)
    self.volume_sdk_client.delete_backup.assert_called_once_with(self.backups[0].id, ignore_missing=False, force=False)