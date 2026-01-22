import operator
import uuid
from osc_placement.tests.functional import base
def test_list_required_trait_any_trait(self):
    rp1 = self.resource_provider_create()
    rp2 = self.resource_provider_create()
    self.resource_inventory_set(rp1['uuid'], 'MEMORY_MB=8192', 'DISK_GB=512')
    self.resource_inventory_set(rp2['uuid'], 'MEMORY_MB=8192', 'DISK_GB=512')
    self.resource_provider_trait_set(rp1['uuid'], 'STORAGE_DISK_SSD', 'HW_NIC_SRIOV_MULTIQUEUE')
    self.resource_provider_trait_set(rp2['uuid'], 'STORAGE_DISK_HDD', 'HW_NIC_SRIOV_MULTIQUEUE')
    rps = self.resource_provider_list(resources=('MEMORY_MB=1024', 'DISK_GB=80'), required=('HW_NIC_SRIOV_MULTIQUEUE',))
    self.assertEqual({rp1['uuid'], rp2['uuid']}, {rp['uuid'] for rp in rps})
    rps = self.resource_provider_list(resources=('MEMORY_MB=1024', 'DISK_GB=80'), required=('STORAGE_DISK_HDD', 'HW_NIC_SRIOV_MULTIQUEUE'))
    self.assertEqual({rp2['uuid']}, {rp['uuid'] for rp in rps})
    rps = self.resource_provider_list(resources=('MEMORY_MB=1024', 'DISK_GB=80'), required=('STORAGE_DISK_HDD,STORAGE_DISK_SSD', 'HW_NIC_SRIOV_MULTIQUEUE'))
    self.assertEqual({rp1['uuid'], rp2['uuid']}, {rp['uuid'] for rp in rps})
    rps = self.resource_provider_list(resources=('MEMORY_MB=1024', 'DISK_GB=80'), required=('STORAGE_DISK_HDD,STORAGE_DISK_SSD', 'HW_NIC_SRIOV_MULTIQUEUE'), forbidden=('STORAGE_DISK_SSD',))
    self.assertEqual({rp2['uuid']}, {rp['uuid'] for rp in rps})