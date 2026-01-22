import collections
import copy
import uuid
from osc_placement.tests.functional import base
def test_inventory_show(self):
    rp_uuid = self.rp['uuid']
    updates = {'min_unit': 1, 'max_unit': 12, 'reserved': 0, 'step_size': 1, 'total': 12, 'allocation_ratio': 16.0}
    expected = updates.copy()
    expected['used'] = 0
    args = ['VCPU:%s=%s' % (k, v) for k, v in updates.items()]
    self.resource_inventory_set(rp_uuid, *args)
    self.assertEqual(expected, self.resource_inventory_show(rp_uuid, 'VCPU', include_used=True))