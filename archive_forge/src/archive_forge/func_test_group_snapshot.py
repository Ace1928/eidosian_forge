from openstack.block_storage.v3 import group as _group
from openstack.block_storage.v3 import group_snapshot as _group_snapshot
from openstack.block_storage.v3 import group_type as _group_type
from openstack.block_storage.v3 import volume as _volume
from openstack.tests.functional.block_storage.v3 import base
def test_group_snapshot(self):
    volume_name = self.getUniqueString()
    self.volume = self.conn.block_storage.create_volume(name=volume_name, volume_type=self.volume_type.id, group_id=self.group.id, size=1)
    self.conn.block_storage.wait_for_status(self.volume, status='available', failures=['error'], interval=2, wait=self._wait_for_timeout)
    self.assertIsInstance(self.volume, _volume.Volume)
    group_snapshot_name = self.getUniqueString()
    self.group_snapshot = self.conn.block_storage.create_group_snapshot(name=group_snapshot_name, group_id=self.group.id)
    self.conn.block_storage.wait_for_status(self.group_snapshot, status='available', failures=['error'], interval=2, wait=self._wait_for_timeout)
    self.assertIsInstance(self.group_snapshot, _group_snapshot.GroupSnapshot)
    group_snapshot = self.conn.block_storage.get_group_snapshot(self.group_snapshot.id)
    self.assertEqual(self.group_snapshot.name, group_snapshot.name)
    group_snapshot = self.conn.block_storage.find_group_snapshot(self.group_snapshot.name)
    self.assertEqual(self.group_snapshot.id, group_snapshot.id)
    group_snapshots = self.conn.block_storage.group_snapshots()
    self.assertIn(self.group_snapshot.id, {g.id for g in group_snapshots})
    self.conn.block_storage.delete_group_snapshot(self.group_snapshot)
    self.conn.block_storage.wait_for_delete(self.group_snapshot)