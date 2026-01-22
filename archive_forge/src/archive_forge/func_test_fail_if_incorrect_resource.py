import collections
import copy
import uuid
from osc_placement.tests.functional import base
def test_fail_if_incorrect_resource(self):
    rp = self.resource_provider_create()
    exc = self.assertRaises(base.CommandException, self.resource_inventory_set, rp['uuid'], 'VCPU')
    self.assertIn('must have "name=value"', str(exc))
    exc = self.assertRaises(base.CommandException, self.resource_inventory_set, rp['uuid'], 'VCPU==')
    self.assertIn('must have "name=value"', str(exc))
    exc = self.assertRaises(base.CommandException, self.resource_inventory_set, rp['uuid'], '=10')
    self.assertIn('must be not empty', str(exc))
    exc = self.assertRaises(base.CommandException, self.resource_inventory_set, rp['uuid'], 'v=')
    self.assertIn('must be not empty', str(exc))
    exc = self.assertRaises(base.CommandException, self.resource_inventory_set, rp['uuid'], 'UNKNOWN_CPU=16')
    self.assertIn('Unknown resource class', str(exc))
    exc = self.assertRaises(base.CommandException, self.resource_inventory_set, rp['uuid'], 'VCPU:fake=16')
    self.assertIn('Unknown inventory field', str(exc))