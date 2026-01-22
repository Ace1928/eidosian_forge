from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_port_create_get_delete(self):
    port = self.create_port(address='11:22:33:44:55:66')
    self.assertEqual(self.node_id, port.node_id)
    self.assertNotEqual(port.is_pxe_enabled, False)
    self.assertIsNone(port.port_group_id)
    loaded = self.conn.baremetal.get_port(port.id)
    self.assertEqual(loaded.id, port.id)
    self.assertIsNotNone(loaded.address)
    with_fields = self.conn.baremetal.get_port(port.id, fields=['uuid', 'extra', 'node_id'])
    self.assertEqual(port.id, with_fields.id)
    self.assertIsNone(with_fields.address)
    self.conn.baremetal.delete_port(port, ignore_missing=False)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_port, port.id)