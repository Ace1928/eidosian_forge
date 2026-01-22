from cinderclient.tests.functional import base
def test_backup_list(self):
    backup_list = self.cinder('backup-list')
    self.assertTableHeaders(backup_list, ['ID', 'Volume ID', 'Status', 'Name', 'Size', 'Object Count', 'Container'])