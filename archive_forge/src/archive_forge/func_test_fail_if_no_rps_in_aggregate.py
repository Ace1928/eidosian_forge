import collections
import copy
import uuid
from osc_placement.tests.functional import base
def test_fail_if_no_rps_in_aggregate(self):
    nonexistent_agg = str(uuid.uuid4())
    exc = self.assertRaises(base.CommandException, self.resource_inventory_set, nonexistent_agg, 'VCPU=8', aggregate=True)
    self.assertIn('No resource providers found in aggregate with uuid {}'.format(nonexistent_agg), str(exc))