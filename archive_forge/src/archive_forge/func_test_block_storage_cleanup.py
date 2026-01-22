import queue
from openstack.tests.functional import base
def test_block_storage_cleanup(self):
    if not self.user_cloud.has_service('object-store'):
        self.skipTest('Object service is requred, but not available')
    status_queue = queue.Queue()
    vol = self.conn.block_storage.create_volume(name='vol1', size='1')
    self.conn.block_storage.wait_for_status(vol)
    s1 = self.conn.block_storage.create_snapshot(volume_id=vol.id)
    self.conn.block_storage.wait_for_status(s1)
    b1 = self.conn.block_storage.create_backup(volume_id=vol.id)
    self.conn.block_storage.wait_for_status(b1)
    b2 = self.conn.block_storage.create_backup(volume_id=vol.id, is_incremental=True, snapshot_id=s1.id)
    self.conn.block_storage.wait_for_status(b2)
    b3 = self.conn.block_storage.create_backup(volume_id=vol.id, is_incremental=True, snapshot_id=s1.id)
    self.conn.block_storage.wait_for_status(b3)
    self.conn.project_cleanup(dry_run=True, wait_timeout=120, status_queue=status_queue, filters={'created_at': '2000-01-01'})
    self.assertTrue(status_queue.empty())
    self.conn.project_cleanup(dry_run=True, wait_timeout=120, status_queue=status_queue, filters={'created_at': '2200-01-01'}, resource_evaluation_fn=lambda x, y, z: False)
    self.assertTrue(status_queue.empty())
    self.conn.project_cleanup(dry_run=True, wait_timeout=120, status_queue=status_queue, filters={'created_at': '2200-01-01'})
    objects = []
    while not status_queue.empty():
        objects.append(status_queue.get())
    volumes = list((obj.id for obj in objects))
    self.assertIn(vol.id, volumes)
    self.conn.project_cleanup(dry_run=True, wait_timeout=120, status_queue=status_queue)
    objects = []
    while not status_queue.empty():
        objects.append(status_queue.get())
    vol_ids = list((obj.id for obj in objects))
    self.assertIn(vol.id, vol_ids)
    vol_check = self.conn.block_storage.get_volume(vol.id)
    self.assertEqual(vol.name, vol_check.name)
    self.conn.project_cleanup(dry_run=False, wait_timeout=600, status_queue=status_queue)
    self.assertEqual(0, len(list(self.conn.block_storage.backups())))
    self.assertEqual(0, len(list(self.conn.block_storage.snapshots())))