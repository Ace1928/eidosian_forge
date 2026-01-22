import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_retired(self):
    reason = "I'm too old for this s...tuff!"
    node = self.create_node()
    node = self.conn.baremetal.update_node(node, is_retired=True)
    self.assertTrue(node.is_retired)
    self.assertIsNone(node.retired_reason)
    node = self.conn.baremetal.get_node(node.id)
    self.assertTrue(node.is_retired)
    self.assertIsNone(node.retired_reason)
    node = self.conn.baremetal.update_node(node, retired_reason=reason)
    self.assertTrue(node.is_retired)
    self.assertEqual(reason, node.retired_reason)
    node = self.conn.baremetal.get_node(node.id)
    self.assertTrue(node.is_retired)
    self.assertEqual(reason, node.retired_reason)
    node = self.conn.baremetal.update_node(node, is_retired=False)
    self.assertFalse(node.is_retired)
    self.assertIsNone(node.retired_reason)
    node = self.conn.baremetal.get_node(node.id)
    self.assertFalse(node.is_retired)
    self.assertIsNone(node.retired_reason)
    node = self.conn.baremetal.update_node(node, is_retired=True, retired_reason=reason)
    self.assertTrue(node.is_retired)
    self.assertEqual(reason, node.retired_reason)
    node = self.conn.baremetal.get_node(node.id)
    self.assertTrue(node.is_retired)
    self.assertEqual(reason, node.retired_reason)