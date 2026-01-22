from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import backup_record
def test_backup_import(self):
    arglist = ['cinder.backup.drivers.swift.SwiftBackupDriver', 'fake_backup_record_data']
    verifylist = [('backup_service', 'cinder.backup.drivers.swift.SwiftBackupDriver'), ('backup_metadata', 'fake_backup_record_data')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, __ = self.cmd.take_action(parsed_args)
    self.backups_mock.import_record.assert_called_with('cinder.backup.drivers.swift.SwiftBackupDriver', 'fake_backup_record_data')
    self.assertEqual(columns, ('backup',))