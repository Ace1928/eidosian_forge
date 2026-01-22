import random
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_allocation_negative_failure(self):
    allocation = self.create_allocation(resource_class=self.resource_class + '-fail')
    self.assertRaises(exceptions.SDKException, self.conn.baremetal.wait_for_allocation, allocation)
    allocation = self.conn.baremetal.get_allocation(allocation.id)
    self.assertEqual('error', allocation.state)
    self.assertIn(self.resource_class + '-fail', allocation.last_error)