import collections
import copy
import uuid
from osc_placement.tests.functional import base
def test_with_aggregate_one_fails(self):
    rps, agg, _invs = self._setup_two_resource_providers_in_aggregate()
    self.resource_class_create('CUSTOM_FOO')
    rp1_uuid = rps[0]['uuid']
    rp1_inv = self.resource_inventory_set(rp1_uuid, 'CUSTOM_FOO=1')
    consumer = str(uuid.uuid4())
    alloc = 'rp=%s,CUSTOM_FOO=1' % rp1_uuid
    self.resource_allocation_set(consumer, [alloc])
    new_resources = ['VCPU:allocation_ratio=5.0', 'VCPU:total=8']
    exc = self.assertRaises(base.CommandException, self.resource_inventory_set, agg, *new_resources, aggregate=True)
    self.assertIn('Failed to set inventory for 1 of 2 resource providers.', str(exc))
    output = self.output.getvalue() + self.error.getvalue()
    self.assertIn('Failed to set inventory for resource provider %s:' % rp1_uuid, output)
    err_txt = "Inventory for 'CUSTOM_FOO' on resource provider '%s' in use." % rp1_uuid
    self.assertIn(err_txt, output)
    placement_defaults = ['VCPU:max_unit=2147483647', 'VCPU:min_unit=1', 'VCPU:reserved=0', 'VCPU:step_size=1']
    new_inventories = self._get_expected_inventories([{}], new_resources + placement_defaults)
    resp = self.resource_inventory_list(rps[1]['uuid'])
    self.assertDictEqual(new_inventories[0], {r['resource_class']: r for r in resp})
    resp = self.resource_inventory_list(rp1_uuid)
    self.assertDictEqual({r['resource_class']: r for r in rp1_inv}, {r['resource_class']: r for r in resp})