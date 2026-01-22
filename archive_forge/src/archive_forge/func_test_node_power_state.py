import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_node_power_state(self):
    node = self.create_node()
    self.assertIsNone(node.power_state)
    self.conn.baremetal.set_node_power_state(node, 'power on', wait=True)
    node = self.conn.baremetal.get_node(node.id)
    self.assertEqual('power on', node.power_state)
    self.conn.baremetal.set_node_power_state(node, 'power off', wait=True)
    node = self.conn.baremetal.get_node(node.id)
    self.assertEqual('power off', node.power_state)