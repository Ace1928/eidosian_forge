import uuid
from osc_placement.tests.functional import base
def test_allocation_unset_resource_classes(self):
    """Tests removing allocations for resource classes."""
    updated_allocs = self.resource_allocation_unset(self.consumer_uuid2, resource_class=['VCPU', 'MEMORY_MB'])
    expected = [{'resource_provider': self.rp3['uuid'], 'generation': 3, 'project_id': self.project_uuid, 'user_id': self.user_uuid, 'resources': {'VGPU': 1}}]
    self.assertEqual(expected, updated_allocs)