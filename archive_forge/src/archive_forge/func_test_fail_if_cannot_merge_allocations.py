import uuid
from osc_lib import exceptions
from oslotest import base
from osc_placement.resources import allocation
def test_fail_if_cannot_merge_allocations(self):
    rp1 = str(uuid.uuid4())
    allocations = ['rp={},VCPU=4,MEMORY_MB=16324'.format(rp1), 'rp={},VCPU=8,DISK_GB=4096'.format(rp1)]
    ex = self.assertRaises(exceptions.CommandError, allocation.parse_allocations, allocations)
    self.assertEqual('Conflict detected for resource provider %s resource class VCPU' % rp1, str(ex))