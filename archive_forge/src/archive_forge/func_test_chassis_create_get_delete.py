from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_chassis_create_get_delete(self):
    chassis = self.create_chassis()
    loaded = self.conn.baremetal.get_chassis(chassis.id)
    self.assertEqual(loaded.id, chassis.id)
    self.conn.baremetal.delete_chassis(chassis, ignore_missing=False)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_chassis, chassis.id)