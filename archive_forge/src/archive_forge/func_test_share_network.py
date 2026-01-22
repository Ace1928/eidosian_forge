from openstack.shared_file_system.v2 import share_network
from openstack.tests.unit import base
def test_share_network(self):
    networks = share_network.ShareNetwork(**EXAMPLE)
    self.assertEqual(EXAMPLE['id'], networks.id)
    self.assertEqual(EXAMPLE['name'], networks.name)
    self.assertEqual(EXAMPLE['project_id'], networks.project_id)
    self.assertEqual(EXAMPLE['description'], networks.description)
    self.assertEqual(EXAMPLE['created_at'], networks.created_at)
    self.assertEqual(EXAMPLE['updated_at'], networks.updated_at)