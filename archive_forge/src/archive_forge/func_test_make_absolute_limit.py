from openstack.block_storage.v2 import limits
from openstack.tests.unit import base
def test_make_absolute_limit(self):
    limit_resource = limits.AbsoluteLimit(**ABSOLUTE_LIMIT)
    self.assertEqual(ABSOLUTE_LIMIT['totalSnapshotsUsed'], limit_resource.total_snapshots_used)
    self.assertEqual(ABSOLUTE_LIMIT['maxTotalBackups'], limit_resource.max_total_backups)
    self.assertEqual(ABSOLUTE_LIMIT['maxTotalVolumeGigabytes'], limit_resource.max_total_volume_gigabytes)
    self.assertEqual(ABSOLUTE_LIMIT['maxTotalSnapshots'], limit_resource.max_total_snapshots)
    self.assertEqual(ABSOLUTE_LIMIT['maxTotalBackupGigabytes'], limit_resource.max_total_backup_gigabytes)
    self.assertEqual(ABSOLUTE_LIMIT['totalBackupGigabytesUsed'], limit_resource.total_backup_gigabytes_used)
    self.assertEqual(ABSOLUTE_LIMIT['maxTotalVolumes'], limit_resource.max_total_volumes)
    self.assertEqual(ABSOLUTE_LIMIT['totalVolumesUsed'], limit_resource.total_volumes_used)
    self.assertEqual(ABSOLUTE_LIMIT['totalBackupsUsed'], limit_resource.total_backups_used)
    self.assertEqual(ABSOLUTE_LIMIT['totalGigabytesUsed'], limit_resource.total_gigabytes_used)