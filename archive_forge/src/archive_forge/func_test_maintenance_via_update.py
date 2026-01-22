import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_maintenance_via_update(self):
    reason = 'Prepating for taking over the world'
    node = self.create_node()
    node = self.conn.baremetal.update_node(node, is_maintenance=True)
    self.assertTrue(node.is_maintenance)
    self.assertIsNone(node.maintenance_reason)
    node = self.conn.baremetal.get_node(node.id)
    self.assertTrue(node.is_maintenance)
    self.assertIsNone(node.maintenance_reason)
    node = self.conn.baremetal.update_node(node, maintenance_reason=reason)
    self.assertTrue(node.is_maintenance)
    self.assertEqual(reason, node.maintenance_reason)
    node = self.conn.baremetal.get_node(node.id)
    self.assertTrue(node.is_maintenance)
    self.assertEqual(reason, node.maintenance_reason)
    node = self.conn.baremetal.update_node(node, is_maintenance=False)
    self.assertFalse(node.is_maintenance)
    self.assertIsNone(node.maintenance_reason)
    node = self.conn.baremetal.get_node(node.id)
    self.assertFalse(node.is_maintenance)
    self.assertIsNone(node.maintenance_reason)
    node = self.conn.baremetal.update_node(node, is_maintenance=True, maintenance_reason=reason)
    self.assertTrue(node.is_maintenance)
    self.assertEqual(reason, node.maintenance_reason)
    node = self.conn.baremetal.get_node(node.id)
    self.assertTrue(node.is_maintenance)
    self.assertEqual(reason, node.maintenance_reason)