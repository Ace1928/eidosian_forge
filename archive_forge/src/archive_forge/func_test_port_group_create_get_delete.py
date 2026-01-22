from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_port_group_create_get_delete(self):
    port_group = self.create_port_group()
    loaded = self.conn.baremetal.get_port_group(port_group.id)
    self.assertEqual(loaded.id, port_group.id)
    self.assertIsNotNone(loaded.node_id)
    with_fields = self.conn.baremetal.get_port_group(port_group.id, fields=['uuid', 'extra'])
    self.assertEqual(port_group.id, with_fields.id)
    self.assertIsNone(with_fields.node_id)
    self.conn.baremetal.delete_port_group(port_group, ignore_missing=False)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_port_group, port_group.id)