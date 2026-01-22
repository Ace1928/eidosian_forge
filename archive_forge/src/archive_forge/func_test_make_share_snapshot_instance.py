from openstack.shared_file_system.v2 import share_snapshot_instance
from openstack.tests.unit import base
def test_make_share_snapshot_instance(self):
    instance = share_snapshot_instance.ShareSnapshotInstance(**EXAMPLE)
    self.assertEqual(EXAMPLE['id'], instance.id)
    self.assertEqual(EXAMPLE['share_id'], instance.share_id)
    self.assertEqual(EXAMPLE['share_instance_id'], instance.share_instance_id)
    self.assertEqual(EXAMPLE['snapshot_id'], instance.snapshot_id)
    self.assertEqual(EXAMPLE['status'], instance.status)
    self.assertEqual(EXAMPLE['progress'], instance.progress)
    self.assertEqual(EXAMPLE['created_at'], instance.created_at)
    self.assertEqual(EXAMPLE['updated_at'], instance.updated_at)
    self.assertEqual(EXAMPLE['provider_location'], instance.provider_location)