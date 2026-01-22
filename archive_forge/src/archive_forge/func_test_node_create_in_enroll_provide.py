import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_node_create_in_enroll_provide(self):
    node = self.create_node()
    self.node_id = node.id
    self.assertEqual(node.driver, 'fake-hardware')
    self.assertEqual(node.provision_state, 'enroll')
    self.assertIsNone(node.power_state)
    self.assertFalse(node.is_maintenance)
    self.conn.baremetal.set_node_provision_state(node, 'manage', wait=True)
    self.assertEqual(node.provision_state, 'manageable')
    self.conn.baremetal.set_node_provision_state(node, 'provide', wait=True)
    self.assertEqual(node.provision_state, 'available')