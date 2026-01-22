from openstack.block_storage.v3 import backup as _backup
from openstack.block_storage.v3 import volume as _volume
from openstack.tests.functional.block_storage.v3 import base
def test_create_metadata(self):
    metadata_backup = self.user_cloud.block_storage.create_backup(name=self.getUniqueString(), volume_id=self.VOLUME_ID, metadata=dict(foo='bar'))
    self.user_cloud.block_storage.wait_for_status(metadata_backup, status='available', failures=['error'], interval=5, wait=self._wait_for_timeout)
    self.user_cloud.block_storage.delete_backup(metadata_backup.id, ignore_missing=False)