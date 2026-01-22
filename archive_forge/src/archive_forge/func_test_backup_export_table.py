from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import backup_record
def test_backup_export_table(self):
    arglist = [self.new_backup.name]
    verifylist = [('backup', self.new_backup.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    parsed_args.formatter = 'table'
    columns, __ = self.cmd.take_action(parsed_args)
    self.backups_mock.export_record.assert_called_with(self.new_backup.id)
    expected_columns = ('Backup Service', 'Metadata')
    self.assertEqual(columns, expected_columns)