import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_node_update_by_name(self):
    self.create_node(name='node-name', extra={'foo': 'bar'})
    instance_uuid = str(uuid.uuid4())
    node = self.conn.baremetal.update_node('node-name', instance_id=instance_uuid, extra={'answer': 42})
    self.assertEqual({'answer': 42}, node.extra)
    self.assertEqual(instance_uuid, node.instance_id)
    node = self.conn.baremetal.get_node('node-name')
    self.assertEqual({'answer': 42}, node.extra)
    self.assertEqual(instance_uuid, node.instance_id)
    node = self.conn.baremetal.update_node('node-name', instance_id=None)
    self.assertIsNone(node.instance_id)
    node = self.conn.baremetal.get_node('node-name')
    self.assertIsNone(node.instance_id)