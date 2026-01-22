from openstack.shared_file_system.v2 import share_snapshot
from openstack.tests.unit import base
def test_make_share_snapshot(self):
    snapshot_resource = share_snapshot.ShareSnapshot(**EXAMPLE)
    self.assertEqual(EXAMPLE['id'], snapshot_resource.id)
    self.assertEqual(EXAMPLE['share_id'], snapshot_resource.share_id)
    self.assertEqual(EXAMPLE['user_id'], snapshot_resource.user_id)
    self.assertEqual(EXAMPLE['created_at'], snapshot_resource.created_at)
    self.assertEqual(EXAMPLE['status'], snapshot_resource.status)
    self.assertEqual(EXAMPLE['name'], snapshot_resource.name)
    self.assertEqual(EXAMPLE['description'], snapshot_resource.description)
    self.assertEqual(EXAMPLE['share_proto'], snapshot_resource.share_proto)
    self.assertEqual(EXAMPLE['share_size'], snapshot_resource.share_size)
    self.assertEqual(EXAMPLE['project_id'], snapshot_resource.project_id)
    self.assertEqual(EXAMPLE['size'], snapshot_resource.size)