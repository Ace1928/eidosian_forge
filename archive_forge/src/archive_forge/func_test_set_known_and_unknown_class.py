import collections
import copy
import uuid
from osc_placement.tests.functional import base
def test_set_known_and_unknown_class(self):
    rp = self.resource_provider_create()
    exc = self.assertRaises(base.CommandException, self.resource_inventory_set, rp['uuid'], 'VCPU=8', 'UNKNOWN=4')
    self.assertIn('Unknown resource class', str(exc))
    self.assertEqual([], self.resource_inventory_list(rp['uuid']))