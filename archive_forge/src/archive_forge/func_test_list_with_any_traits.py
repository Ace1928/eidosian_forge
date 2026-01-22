import uuid
from osc_placement.tests.functional import base
def test_list_with_any_traits(self):
    groups = {'': {'resources': ('DISK_GB=1',), 'required': ('STORAGE_DISK_HDD',)}, '1': {'resources': ('VCPU=1',), 'required': ('HW_CPU_X86_AVX',)}}
    rows = self.allocation_candidate_granular(groups=groups)
    numbers = {row['#'] for row in rows}
    self.assertEqual(1, len(numbers))
    self.assertEqual(2, len(rows))
    rps = {row['resource provider'] for row in rows}
    self.assertEqual({self.rp1['uuid'], self.rp1_1['uuid']}, rps)
    groups = {'': {'resources': ('DISK_GB=1',), 'required': ('STORAGE_DISK_HDD,STORAGE_DISK_SSD',)}, '1': {'resources': ('VCPU=1',), 'required': ('HW_CPU_X86_AVX',)}}
    rows = self.allocation_candidate_granular(groups=groups)
    numbers = {row['#'] for row in rows}
    self.assertEqual(2, len(numbers))
    self.assertEqual(4, len(rows))
    rps = {row['resource provider'] for row in rows}
    self.assertEqual({self.rp1['uuid'], self.rp1_1['uuid'], self.rp2['uuid'], self.rp2_1['uuid']}, rps)
    groups = {'': {'resources': ('DISK_GB=1',), 'required': ('STORAGE_DISK_HDD,STORAGE_DISK_SSD', 'STORAGE_DISK_SSD'), 'forbidden': ('STORAGE_DISK_HDD',)}, '1': {'resources': ('VCPU=1',), 'required': ('HW_CPU_X86_AVX,HW_CPU_X86_SSE',), 'forbidden': ('HW_CPU_X86_SSE',)}}
    rows = self.allocation_candidate_granular(groups=groups)
    numbers = {row['#'] for row in rows}
    self.assertEqual(1, len(numbers))
    self.assertEqual(2, len(rows))
    rps = {row['resource provider'] for row in rows}
    self.assertEqual({self.rp2['uuid'], self.rp2_1['uuid']}, rps)