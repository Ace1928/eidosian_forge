import random
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_allocation_patch(self):
    name = 'ossdk-name2'
    allocation = self.create_allocation(resource_class=self.resource_class)
    allocation = self.conn.baremetal.wait_for_allocation(allocation)
    self.assertEqual('active', allocation.state)
    self.assertIsNone(allocation.last_error)
    self.assertIsNone(allocation.name)
    self.assertEqual({}, allocation.extra)
    allocation = self.conn.baremetal.patch_allocation(allocation, [{'op': 'replace', 'path': '/name', 'value': name}, {'op': 'add', 'path': '/extra/answer', 'value': 42}])
    self.assertEqual(name, allocation.name)
    self.assertEqual({'answer': 42}, allocation.extra)
    allocation = self.conn.baremetal.get_allocation(name)
    self.assertEqual(name, allocation.name)
    self.assertEqual({'answer': 42}, allocation.extra)
    allocation = self.conn.baremetal.patch_allocation(allocation, [{'op': 'remove', 'path': '/name'}, {'op': 'remove', 'path': '/extra/answer'}])
    self.assertIsNone(allocation.name)
    self.assertEqual({}, allocation.extra)
    allocation = self.conn.baremetal.get_allocation(allocation.id)
    self.assertIsNone(allocation.name)
    self.assertEqual({}, allocation.extra)
    self.conn.baremetal.delete_allocation(allocation, ignore_missing=False)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_allocation, allocation.id)