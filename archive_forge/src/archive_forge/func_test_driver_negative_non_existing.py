from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_driver_negative_non_existing(self):
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_driver, 'not-a-driver')